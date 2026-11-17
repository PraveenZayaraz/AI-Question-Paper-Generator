"""Data shapes shared by the pipeline, the API and the CLI."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Topic(BaseModel):
    name: str
    # Number of retrieved chunks above the similarity threshold: how much
    # textbook material backs this topic. Zero means "not covered by the book".
    support: int = 0
    best_score: float = 0.0


class Unit(BaseModel):
    number: int
    title: str
    topics: List[Topic] = Field(default_factory=list)


class Syllabus(BaseModel):
    units: List[Unit] = Field(default_factory=list)
    parser: str = "heuristic"  # "heuristic" or "llm"

    def find_topic(self, name: str):
        wanted = name.strip().lower()
        for unit in self.units:
            for topic in unit.topics:
                if topic.name.lower() == wanted:
                    return unit, topic
        return None


class Chunk(BaseModel):
    id: int
    text: str
    source: str
    page_start: int
    page_end: int


class SubjectMeta(BaseModel):
    id: str
    name: str
    status: str = "ingesting"  # ingesting | ready | failed
    message: str = ""
    created_at: str = ""
    n_chunks: int = 0
    sources: List[str] = Field(default_factory=list)
    embedder: str = ""
    syllabus: Optional[Syllabus] = None


class SourceRef(BaseModel):
    chunk_id: int
    source: str
    page_start: int
    page_end: int
    score: float


class Question(BaseModel):
    text: str
    marks: int
    bloom: str
    verb: str
    unit: int
    topic: str
    sources: List[SourceRef] = Field(default_factory=list)
    # Cosine similarity between the question and its best source chunk.
    grounding: float = 0.0
    latency_ms: int = 0
    attempts: int = 1


class SectionSpec(BaseModel):
    name: str
    marks: int = Field(gt=0, le=100)
    count: int = Field(gt=0, le=50)
    either_or: bool = False
    split: bool = False
    bloom: List[str] = Field(default_factory=list)


class Blueprint(BaseModel):
    title: str = "Question Paper"
    duration_minutes: Optional[int] = None
    sections: List[SectionSpec]

    @property
    def total_marks(self) -> int:
        return sum(s.marks * s.count for s in self.sections)


class PaperItem(BaseModel):
    number: int
    # Each option is a list of sub-questions. Two options means "either / or".
    options: List[List[Question]]


class PaperSection(BaseModel):
    name: str
    marks: int
    items: List[PaperItem] = Field(default_factory=list)


class Paper(BaseModel):
    id: str
    subject_id: str
    subject_name: str
    status: str = "generating"  # generating | ready | failed
    message: str = ""
    progress: int = 0
    total: int = 0
    created_at: str = ""
    blueprint: Blueprint
    sections: List[PaperSection] = Field(default_factory=list)
