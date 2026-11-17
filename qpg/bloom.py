"""Bloom's taxonomy: the cognitive levels that set how demanding a question is."""
from __future__ import annotations

from typing import List

BLOOM_VERBS = {
    "Remember": ["define", "list", "state", "name", "identify"],
    "Understand": ["explain", "describe", "summarize", "illustrate", "interpret"],
    "Apply": ["apply", "demonstrate", "solve", "implement", "show how"],
    "Analyze": ["analyze", "compare", "differentiate", "examine", "contrast"],
    "Evaluate": ["evaluate", "justify", "critique", "assess", "defend"],
    "Create": ["design", "propose", "formulate", "construct", "develop"],
}
LEVELS = list(BLOOM_VERBS)


def levels_for_marks(marks: int) -> List[str]:
    """Low-mark questions test recall; high-mark questions test reasoning."""
    if marks <= 2:
        return ["Remember", "Understand"]
    if marks <= 5:
        return ["Understand", "Apply"]
    if marks <= 9:
        return ["Apply", "Analyze"]
    return ["Analyze", "Evaluate", "Create"]


def normalize_levels(levels, marks: int) -> List[str]:
    known = {level.lower(): level for level in LEVELS}
    picked = [known[str(name).lower()] for name in (levels or []) if str(name).lower() in known]
    return picked or levels_for_marks(marks)


def answer_guidance(marks: int) -> str:
    if marks <= 2:
        return "answerable in one or two sentences"
    if marks <= 5:
        return "answerable in a short paragraph"
    if marks <= 9:
        return "needs a structured answer with several points"
    return "needs a detailed, well-argued answer"
