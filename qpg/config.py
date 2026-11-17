"""Runtime settings, read once from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(f"QPG_{name}", default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "data")).resolve())
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://localhost:11434"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "llama3"))
    llm_timeout: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT", "120")))
    # "st" = sentence-transformers, "hashing" = dependency-free lexical fallback
    embedder: str = field(default_factory=lambda: _env("EMBEDDER", "st"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "all-MiniLM-L6-v2"))
    chunk_chars: int = field(default_factory=lambda: int(_env("CHUNK_CHARS", "900")))
    # A topic counts as covered when a chunk scores at least this cosine similarity.
    min_similarity: float = field(default_factory=lambda: float(_env("MIN_SIMILARITY", "0.35")))
    context_chunks: int = field(default_factory=lambda: int(_env("CONTEXT_CHUNKS", "3")))
    # A passage is used only if it scores at least this share of the best passage's similarity.
    focus_ratio: float = field(default_factory=lambda: float(_env("FOCUS_RATIO", "0.8")))
    # Passages for one question must start within this many pages of each other.
    neighbor_pages: int = field(default_factory=lambda: int(_env("NEIGHBOR_PAGES", "8")))
    # A question scoring below this against its sources is retried.
    min_grounding: float = field(default_factory=lambda: float(_env("MIN_GROUNDING", "0.45")))
    # Two questions closer than this cosine similarity are treated as duplicates.
    duplicate_similarity: float = field(default_factory=lambda: float(_env("DUPLICATE_SIMILARITY", "0.88")))
    max_attempts: int = field(default_factory=lambda: int(_env("MAX_ATTEMPTS", "3")))
    max_upload_mb: int = field(default_factory=lambda: int(_env("MAX_UPLOAD_MB", "200")))
    cors_origins: tuple = field(
        default_factory=lambda: tuple(
            o.strip() for o in _env("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()
        )
    )

    @property
    def subjects_dir(self) -> Path:
        return self.data_dir / "subjects"

    @property
    def papers_dir(self) -> Path:
        return self.data_dir / "papers"
