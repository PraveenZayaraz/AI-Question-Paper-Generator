"""Application service: wires settings, storage, the embedder and the language
model together. The API and the CLI both go through this one class."""
from __future__ import annotations

import logging
import random
import threading
from pathlib import Path
from typing import List, Optional, Sequence

from qpg.config import Settings
from qpg.embeddings import make_embedder
from qpg.generate import NoSourceMaterial, QuestionGenerator
from qpg.ingest import ingest_subject
from qpg.llm import OllamaClient
from qpg.models import Blueprint, Paper, Question, SubjectMeta
from qpg.paper import TopicPicker, build_paper, default_blueprint, export_docx
from qpg.store import Store, now_iso

log = logging.getLogger("qpg")


class Service:
    def __init__(self, settings: Optional[Settings] = None, embedder=None, llm=None):
        self.settings = settings or Settings()
        self.store = Store(self.settings.subjects_dir, self.settings.papers_dir)
        self._embedder = embedder
        self._embedder_lock = threading.Lock()
        self.llm = llm or OllamaClient(self.settings.ollama_url, self.settings.llm_model, self.settings.llm_timeout)

    @property
    def embedder(self):
        # Loaded once, on first use, and shared by every request.
        with self._embedder_lock:
            if self._embedder is None:
                self._embedder = make_embedder(self.settings.embedder, self.settings.embed_model)
            return self._embedder

    # ---- ingestion ------------------------------------------------------
    def ingest(
        self,
        subject_id: str,
        textbooks: Sequence[Path],
        syllabus_path: Optional[Path] = None,
        syllabus_text: str = "",
        progress=None,
    ) -> SubjectMeta:
        llm = self.llm if self.llm.available() else None  # syllabus parsing works without it
        return ingest_subject(
            self.store, subject_id, textbooks, self.embedder, self.settings,
            syllabus_path=syllabus_path, syllabus_text=syllabus_text, llm=llm, progress=progress,
        )

    def ingest_in_background(self, subject_id: str, **kwargs) -> None:
        def run():
            try:
                self.ingest(subject_id, **kwargs)
            except Exception:
                log.exception("Ingestion failed for subject %s", subject_id)  # also recorded on the subject

        threading.Thread(target=run, daemon=True).start()

    # ---- questions ------------------------------------------------------
    def generator(self, subject_id: str, seed: Optional[int] = None) -> QuestionGenerator:
        subject = self.store.load_subject(subject_id, self.embedder)
        return QuestionGenerator(subject, self.llm, self.embedder, self.settings, random.Random(seed))

    def generate_questions(
        self,
        subject_id: str,
        marks: int,
        count: int,
        unit: Optional[int] = None,
        topic: Optional[str] = None,
        bloom: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> List[Question]:
        gen = self.generator(subject_id, seed)
        syllabus = gen.subject.meta.syllabus
        questions: List[Question] = []
        if topic:
            unit_no, name = gen.resolve_topic(topic)
            for _ in range(count):
                questions.append(gen.generate(name, marks, unit=unit_no, bloom_levels=bloom))
            return questions

        units = [u for u in syllabus.units if unit is None or u.number == unit]
        if not units:
            raise ValueError(f"The syllabus has no unit {unit}.")
        picker = TopicPicker(syllabus, gen.rng)
        attempts = 0
        while len(questions) < count and attempts < count * 4:
            current = units[attempts % len(units)]
            attempts += 1
            name = picker.pick(current)
            if name is None:
                continue
            try:
                questions.append(gen.generate(name, marks, unit=current.number, bloom_levels=bloom))
            except NoSourceMaterial:
                continue
        if not questions:
            raise ValueError("The uploaded material does not cover the selected unit.")
        return questions

    # ---- papers ---------------------------------------------------------
    def default_blueprint(self, subject_id: str) -> Blueprint:
        meta = self.store.get_meta(subject_id)
        return default_blueprint(len(meta.syllabus.units) if meta.syllabus else 1)

    def create_paper(self, subject_id: str, blueprint: Optional[Blueprint] = None) -> Paper:
        meta = self.store.get_meta(subject_id)
        if meta.status != "ready":
            raise ValueError(f"Subject '{meta.name}' is not ready yet.")
        paper = Paper(
            id=self.store.new_paper_id(), subject_id=subject_id, subject_name=meta.name,
            created_at=now_iso(), blueprint=blueprint or self.default_blueprint(subject_id),
        )
        self.store.save_paper(paper)
        return paper

    def run_paper(self, paper: Paper, seed: Optional[int] = None) -> Paper:
        try:
            gen = self.generator(paper.subject_id, seed)
            build_paper(paper, gen.subject.meta.syllabus, gen, on_progress=self.store.save_paper)
            export_docx(paper, self.store.paper_docx(paper.id))
        except Exception as exc:
            paper.status = "failed"
            paper.message = str(exc)
            self.store.save_paper(paper)
            raise
        self.store.save_paper(paper)
        return paper

    def run_paper_in_background(self, paper: Paper) -> None:
        def run():
            try:
                self.run_paper(paper)
            except Exception:
                log.exception("Paper %s failed", paper.id)  # also recorded on the paper

        threading.Thread(target=run, daemon=True).start()
