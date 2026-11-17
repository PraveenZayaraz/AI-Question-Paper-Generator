"""On-disk storage. Each subject is one folder:

    data/subjects/<subject-id>/
        meta.json        name, status, syllabus
        chunks.jsonl     one textbook chunk per line
        embeddings.npy   one vector per chunk, same order
        uploads/         the original files
"""
from __future__ import annotations

import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from qpg.index import HybridRetriever
from qpg.models import Chunk, Paper, SubjectMeta

_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,80}$")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "subject"
    return f"{slug}-{uuid.uuid4().hex[:6]}"


def _check_id(value: str) -> str:
    # Ids become folder and file names, so anything that could escape the data
    # directory ("..", slashes, drive letters) is rejected here.
    if not _ID.match(value or ""):
        raise KeyError(f"Invalid id: {value!r}")
    return value


# One lock for every JSON read and write. A background job rewrites a paper's
# file while the API polls it; on Windows, replacing a file that another thread
# has open raises PermissionError, so reads and writes must not overlap.
_io_lock = threading.RLock()


def _read_text(path: Path) -> str:
    with _io_lock:
        return path.read_text(encoding="utf-8")


def _write_json(path: Path, payload: str) -> None:
    with _io_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        for attempt in range(10):
            try:
                tmp.replace(path)  # atomic: readers never see a half-written file
                return
            except PermissionError:
                # Another program (antivirus, a file indexer) may hold the file briefly.
                if attempt == 9:
                    raise
                time.sleep(0.05)


class LoadedSubject:
    def __init__(self, meta: SubjectMeta, chunks: List[Chunk], vectors: np.ndarray, embedder):
        self.meta = meta
        self.chunks = chunks
        self.vectors = vectors
        self.retriever = HybridRetriever(chunks, vectors, embedder)


class Store:
    def __init__(self, subjects_dir: Path, papers_dir: Path):
        self.subjects_dir = Path(subjects_dir)
        self.papers_dir = Path(papers_dir)
        self._cache: Dict[str, LoadedSubject] = {}
        self._lock = threading.Lock()

    # ---- subjects -------------------------------------------------------
    def subject_dir(self, subject_id: str) -> Path:
        return self.subjects_dir / _check_id(subject_id)

    def create_subject(self, name: str) -> SubjectMeta:
        meta = SubjectMeta(id=slugify(name), name=name.strip() or "Untitled subject", created_at=now_iso())
        (self.subject_dir(meta.id) / "uploads").mkdir(parents=True, exist_ok=True)
        self.save_meta(meta)
        return meta

    def save_meta(self, meta: SubjectMeta) -> None:
        _write_json(self.subject_dir(meta.id) / "meta.json", meta.model_dump_json(indent=2))
        with self._lock:
            self._cache.pop(meta.id, None)

    def get_meta(self, subject_id: str) -> SubjectMeta:
        path = self.subject_dir(subject_id) / "meta.json"
        if not path.exists():
            raise KeyError(f"No subject with id '{subject_id}'.")
        return SubjectMeta.model_validate_json(_read_text(path))

    def list_subjects(self) -> List[SubjectMeta]:
        if not self.subjects_dir.exists():
            return []
        metas = []
        for folder in sorted(self.subjects_dir.iterdir()):
            if (folder / "meta.json").exists() and _ID.match(folder.name):
                metas.append(self.get_meta(folder.name))
        return sorted(metas, key=lambda m: m.created_at, reverse=True)

    def delete_subject(self, subject_id: str) -> None:
        import shutil

        folder = self.subject_dir(subject_id)
        if not folder.exists():
            raise KeyError(f"No subject with id '{subject_id}'.")
        shutil.rmtree(folder)
        with self._lock:
            self._cache.pop(subject_id, None)

    def save_corpus(self, subject_id: str, chunks: List[Chunk], vectors: np.ndarray) -> None:
        folder = self.subject_dir(subject_id)
        with open(folder / "chunks.jsonl", "w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(chunk.model_dump_json() + "\n")
        np.save(folder / "embeddings.npy", np.asarray(vectors, dtype=np.float32))

    def load_corpus(self, subject_id: str):
        folder = self.subject_dir(subject_id)
        chunks = []
        with open(folder / "chunks.jsonl", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    chunks.append(Chunk.model_validate_json(line))
        vectors = np.load(folder / "embeddings.npy")
        return chunks, vectors

    def load_subject(self, subject_id: str, embedder) -> LoadedSubject:
        """Load a subject and build its search index once; later calls reuse it."""
        with self._lock:
            cached = self._cache.get(subject_id)
        if cached is not None:
            return cached
        meta = self.get_meta(subject_id)
        if meta.status != "ready":
            raise ValueError(f"Subject '{meta.name}' is not ready (status: {meta.status}). {meta.message}".strip())
        if meta.embedder and meta.embedder != embedder.name:
            raise ValueError(
                f"Subject '{meta.name}' was indexed with '{meta.embedder}' but the server is using "
                f"'{embedder.name}'. Re-ingest the subject or switch QPG_EMBEDDER back."
            )
        chunks, vectors = self.load_corpus(subject_id)
        loaded = LoadedSubject(meta, chunks, vectors, embedder)
        with self._lock:
            self._cache[subject_id] = loaded
        return loaded

    # ---- papers ---------------------------------------------------------
    def paper_json(self, paper_id: str) -> Path:
        return self.papers_dir / f"{_check_id(paper_id)}.json"

    def paper_docx(self, paper_id: str) -> Path:
        return self.papers_dir / f"{_check_id(paper_id)}.docx"

    def new_paper_id(self) -> str:
        return f"paper-{uuid.uuid4().hex[:10]}"

    def save_paper(self, paper: Paper) -> None:
        _write_json(self.paper_json(paper.id), paper.model_dump_json(indent=2))

    def get_paper(self, paper_id: str) -> Paper:
        path = self.paper_json(paper_id)
        if not path.exists():
            raise KeyError(f"No paper with id '{paper_id}'.")
        return Paper.model_validate_json(_read_text(path))

    def list_papers(self, subject_id: Optional[str] = None) -> List[Paper]:
        if not self.papers_dir.exists():
            return []
        files = self.papers_dir.glob("paper-*.json")
        papers = [Paper.model_validate_json(_read_text(f)) for f in files]
        if subject_id:
            papers = [p for p in papers if p.subject_id == subject_id]
        return sorted(papers, key=lambda p: p.created_at, reverse=True)
