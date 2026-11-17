"""Split page text into overlapping, sentence-aligned chunks that remember their pages."""
from __future__ import annotations

import re
from typing import List, Tuple

from qpg.models import Chunk

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


def clean_text(text: str) -> str:
    text = text.replace("­", "")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)  # re-join words hyphenated across lines
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str, max_chars: int) -> List[str]:
    sentences = []
    for sentence in _SENTENCE_END.split(text):
        sentence = sentence.strip()
        while len(sentence) > max_chars:  # tables and code can lack sentence breaks
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            sentences.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            sentences.append(sentence)
    return sentences


_NUMBERED_ITEM = re.compile(r"(?:^|\s)\d{1,2}\.\s+[A-Z]")
_CITATION = re.compile(r"\bpp\.\s*\d|\bet al\.|\bvol\.\s*\d", re.I)


def is_back_matter(text: str) -> bool:
    """End-of-chapter problem sets and reference lists.

    A chunk made of several numbered items ("12. A router has ... 13. Compute ...")
    is an exercise list; one full of "pp. 12-19" and "et al." is a bibliography.
    Neither explains the subject, and questions written from them copy the
    textbook's own exercises.
    """
    return len(_NUMBERED_ITEM.findall(text)) >= 3 or len(_CITATION.findall(text)) >= 3


def chunk_pages(
    pages: List[Tuple[int, str]],
    source: str,
    max_chars: int = 900,
    overlap_sentences: int = 1,
    min_words: int = 30,
    start_id: int = 0,
) -> List[Chunk]:
    """Pack sentences into chunks of at most `max_chars`.

    The last `overlap_sentences` sentences of a chunk are repeated at the start
    of the next one, so an idea that straddles a boundary is whole in at least
    one chunk.
    """
    sentences: List[Tuple[int, str]] = []
    for number, text in pages:
        for sentence in split_sentences(clean_text(text), max_chars):
            sentences.append((number, sentence))

    chunks: List[Chunk] = []
    current: List[Tuple[int, str]] = []
    size = 0

    def flush():
        nonlocal current, size
        if not current:
            return
        text = " ".join(s for _, s in current)
        if len(text.split()) >= min_words and not is_back_matter(text):
            chunks.append(
                Chunk(
                    id=start_id + len(chunks),
                    text=text,
                    source=source,
                    page_start=current[0][0],
                    page_end=current[-1][0],
                )
            )
        tail = current[-overlap_sentences:] if overlap_sentences else []
        current = list(tail)
        size = sum(len(s) + 1 for _, s in current)

    fresh = 0  # sentences added since the last flush, so overlap alone never forms a chunk
    for number, sentence in sentences:
        if current and size + len(sentence) + 1 > max_chars and fresh:
            flush()
            fresh = 0
        current.append((number, sentence))
        size += len(sentence) + 1
        fresh += 1
    if fresh:
        flush()
    return chunks
