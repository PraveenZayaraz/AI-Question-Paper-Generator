"""Ingestion: files in, searchable subject out.

    textbooks -> pages -> cleaned chunks -> embeddings -> saved corpus
    syllabus  -> units and topics -> coverage check against the corpus
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np

from qpg.chunking import chunk_pages
from qpg.config import Settings
from qpg.extract import extract_pages, extract_text, strip_repeated_lines
from qpg.index import HybridRetriever
from qpg.models import Chunk, Syllabus
from qpg.store import Store
from qpg.syllabus import parse_syllabus, topic_query

Progress = Callable[[str], None]


def build_corpus(paths: Sequence[Path], settings: Settings, progress: Progress) -> List[Chunk]:
    chunks: List[Chunk] = []
    for path in paths:
        progress(f"Reading {Path(path).name}")
        pages = strip_repeated_lines(extract_pages(path))
        chunks.extend(
            chunk_pages(pages, source=Path(path).name, max_chars=settings.chunk_chars, start_id=len(chunks))
        )
    return chunks


def embed_chunks(chunks: List[Chunk], embedder, progress: Progress, batch: int = 256) -> np.ndarray:
    parts = []
    for start in range(0, len(chunks), batch):
        progress(f"Embedding chunks {start + 1}-{min(start + batch, len(chunks))} of {len(chunks)}")
        parts.append(embedder.encode([c.text for c in chunks[start : start + batch]]))
    return np.vstack(parts) if parts else np.zeros((0, 1), dtype=np.float32)


def score_coverage(syllabus: Syllabus, retriever: HybridRetriever, min_similarity: float, k: int = 8) -> None:
    """Record, for every topic, how much textbook material backs it.

    Topics with no chunk above the threshold are kept but flagged (support = 0),
    so the generator can skip them instead of inventing questions.
    """
    for unit in syllabus.units:
        for topic in unit.topics:
            hits = retriever.search(topic.name, k=k, context_query=topic_query(topic.name, unit.title))
            topic.best_score = round(max((h.similarity for h in hits), default=0.0), 4)
            topic.support = sum(h.similarity >= min_similarity for h in hits)


def ingest_subject(
    store: Store,
    subject_id: str,
    textbook_paths: Sequence[Path],
    embedder,
    settings: Settings,
    syllabus_path: Optional[Path] = None,
    syllabus_text: str = "",
    llm=None,
    progress: Optional[Progress] = None,
):
    meta = store.get_meta(subject_id)

    def report(message: str) -> None:
        meta.message = message
        store.save_meta(meta)
        if progress:
            progress(message)

    try:
        if not textbook_paths:
            raise ValueError("Add at least one textbook or notes file.")
        text = syllabus_text or (extract_text(syllabus_path) if syllabus_path else "")
        if not text.strip():
            raise ValueError("The syllabus is empty or could not be read.")

        chunks = build_corpus(textbook_paths, settings, report)
        if not chunks:
            raise ValueError(
                "No readable text was found in the textbook. Scanned PDFs need OCR before they can be used."
            )
        vectors = embed_chunks(chunks, embedder, report)
        store.save_corpus(subject_id, chunks, vectors)

        report("Reading the syllabus")
        syllabus = parse_syllabus(text, llm=llm)
        if not syllabus.units:
            raise ValueError("No units or topics could be found in the syllabus.")
        report("Matching syllabus topics to the textbook")
        score_coverage(syllabus, HybridRetriever(chunks, vectors, embedder), settings.min_similarity)

        meta.syllabus = syllabus
        meta.n_chunks = len(chunks)
        meta.sources = [Path(p).name for p in textbook_paths]
        meta.embedder = embedder.name
        meta.status = "ready"
        meta.message = ""
        store.save_meta(meta)
    except Exception as exc:
        meta.status = "failed"
        meta.message = str(exc)
        store.save_meta(meta)
        raise
    return meta
