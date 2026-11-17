"""Hybrid retrieval: dense vectors (FAISS) + BM25 keywords, merged with reciprocal rank fusion."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from qpg.models import Chunk

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will with".split()
)


def tokenize(text: str) -> List[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS]


class VectorIndex:
    """Exact inner-product search. Uses FAISS when installed, NumPy otherwise.

    Vectors are normalised, so inner product equals cosine similarity.
    """

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._faiss = None
        if len(self.vectors):
            try:
                import faiss

                self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
                self._faiss.add(self.vectors)
            except ImportError:
                self._faiss = None

    @property
    def backend(self) -> str:
        return "faiss" if self._faiss is not None else "numpy"

    def search(self, query: np.ndarray, k: int):
        k = min(k, len(self.vectors))
        if k == 0:
            return [], []
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if self._faiss is not None:
            scores, ids = self._faiss.search(query, k)
            return scores[0].tolist(), ids[0].tolist()
        scores = self.vectors @ query[0]
        ids = np.argsort(-scores)[:k]
        return scores[ids].tolist(), ids.tolist()


class BM25:
    """Okapi BM25 keyword ranking."""

    def __init__(self, documents: Sequence[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_len = [len(d) for d in documents]
        self.avg_len = (sum(self.doc_len) / len(documents)) if documents else 0.0
        self.postings: Dict[str, List[tuple]] = {}
        for doc_id, tokens in enumerate(documents):
            for term, freq in Counter(tokens).items():
                self.postings.setdefault(term, []).append((doc_id, freq))
        n = len(documents)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def search(self, query_tokens: Sequence[str], k: int):
        scores: Dict[int, float] = {}
        for term in set(query_tokens):
            for doc_id, freq in self.postings.get(term, ()):
                norm = freq + self.k1 * (1 - self.b + self.b * self.doc_len[doc_id] / (self.avg_len or 1))
                scores[doc_id] = scores.get(doc_id, 0.0) + self.idf[term] * freq * (self.k1 + 1) / norm
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [s for _, s in ranked], [i for i, _ in ranked]


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = 60) -> List[int]:
    """Merge ranked lists. An item scores 1/(k + rank) in each list it appears in."""
    fused: Dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return [item for item, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))]


@dataclass
class Hit:
    chunk: Chunk
    similarity: float  # cosine similarity between the query and this chunk
    rank: int


class HybridRetriever:
    def __init__(self, chunks: List[Chunk], vectors: np.ndarray, embedder, candidates: int = 20):
        self.chunks = chunks
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.embedder = embedder
        self.candidates = candidates
        self.dense = VectorIndex(self.vectors)
        self.sparse = BM25([tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int = 5, context_query: str = "", context_below: float = 0.5) -> List[Hit]:
        """Rank chunks for `query`.

        `context_query` is a longer form of the same query (the topic plus its
        unit title). It is used only when the topic alone finds no confident
        match (best similarity under `context_below`). A vague topic such as
        "Design Issues" needs the unit to land in the right chapter, but a
        specific one such as "CSMA Protocols" already points at its own section
        and the unit title would drag it toward the chapter introduction.
        """
        if not self.chunks:
            return []
        queries = [query]
        vectors = self.embedder.encode(queries)
        plain_scores, plain_ids = self.dense.search(vectors[0], self.candidates)
        rankings = [plain_ids, self.sparse.search(tokenize(query), self.candidates)[1]]
        # A one-word topic ("Architecture", "Services") is never specific enough alone.
        specific = len(tokenize(query)) >= 2
        confident = specific and bool(plain_scores) and plain_scores[0] >= context_below
        if context_query and context_query != query and not confident:
            context_vector = self.embedder.encode([context_query])
            vectors = np.vstack([vectors, context_vector])
            rankings.append(self.dense.search(context_vector[0], self.candidates)[1])
            rankings.append(self.sparse.search(tokenize(context_query), self.candidates)[1])
        order = reciprocal_rank_fusion(rankings)[:k]
        return [
            Hit(chunk=self.chunks[i], similarity=float(np.max(vectors @ self.vectors[i])), rank=rank)
            for rank, i in enumerate(order, start=1)
        ]

    def as_hits(self, query: str, chunk_ids: Sequence[int]) -> List[Hit]:
        """Wrap specific chunks as hits, scored against `query`."""
        if not chunk_ids:
            return []
        vector = self.embedder.encode([query])[0]
        return [Hit(chunk=self.chunks[i], similarity=float(self.vectors[i] @ vector), rank=0) for i in chunk_ids]
