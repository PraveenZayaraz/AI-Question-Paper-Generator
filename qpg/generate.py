"""Question generation: retrieve source text for a topic, prompt the model,
then validate, de-duplicate and score what comes back."""
from __future__ import annotations

import random
import re
import time
from typing import Dict, List, Optional

import numpy as np

from qpg.bloom import BLOOM_VERBS, answer_guidance, normalize_levels
from qpg.config import Settings
from qpg.llm import LLMError
from qpg.models import Question, SourceRef
from qpg.store import LoadedSubject
from qpg.syllabus import topic_query

PROMPT = """You write exam questions for a university course.

Topic: {topic}{unit_line}
Marks: {marks} ({guidance})
Bloom's taxonomy level: {bloom}.
Start the question with one of these verbs, whichever fits the material best: {verbs}

Rules:
- Write exactly one question that can be answered from the source material below.
- Keep it to one or two sentences and under 45 words, the way a printed exam question reads.
- The question must make sense on its own. Never refer to "the text", "the passage", "the course material",
  "this unit" or "this chapter", or to anything being "discussed", "studied" or "mentioned" somewhere.
- Do not include the answer, hints, numbering or the marks.
{extra}
Return JSON in exactly this shape: {{"question": "...", "verb": "the verb you used"}}

Source material:
{context}
"""

# Phrases that show the model is pointing at its prompt instead of asking a
# standalone question. Kept narrow on purpose: "the source address" and
# "in the context of" are normal exam wording and must pass.
_LEAKS = re.compile(
    r"\b(passage|excerpt)\b"
    r"|\b(course|source|lecture|provided|given|above|reference|study) (materials?|text|notes|content)\b"
    r"|\baccording to the (text|material|notes|source|book|textbook|author)\b"
    r"|\b(discussed|mentioned|described|stated|explained|covered) (in the|above|earlier|previously)\b"
    r"|\b(in|of|from|within) this (unit|chapter|section|course|book|module|lesson)\b"
    r"|\b(studied|covered|taught|learned|learnt) (in|so far|earlier|above)\b"
    r"|\bthe (text|textbook|author)\b"
    r"|\[\d+\]|\bas (described|shown|given|stated|defined) in\b",  # "[1]", "as described in ..."
    re.I,
)
MAX_WORDS = 50
_PREAMBLE = re.compile(r"^(here is|here's|sure|question\s*:|q\d*[.:)])\s*", re.I)


def clean_question(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip().strip('"').strip()
    text = _PREAMBLE.sub("", text)
    text = re.sub(r"^\(?\d+\s*marks?\)?\s*[:.-]?\s*", "", text, flags=re.I)
    return text.strip()


_SENTENCE_BREAK = re.compile(r"[.?!]\s+(?=[A-Z])")


def validate_question(text: str, marks: Optional[int] = None) -> Optional[str]:
    """Return None when the question is usable, otherwise the reason it is not."""
    if len(text) < 15:
        return "was too short"
    if len(text.split()) > MAX_WORDS:
        return f"was longer than {MAX_WORDS} words"
    if _LEAKS.search(text):
        return "referred to the source material instead of standing on its own"
    if re.search(r"\banswer\s*:", text, flags=re.I):
        return "included an answer"
    if marks is not None and marks <= 2 and _SENTENCE_BREAK.search(text):
        return f"asked more than one thing for a {marks}-mark question"
    return None


def pick_verb(text: str, claimed: str, verbs: List[str]) -> str:
    """Work out which Bloom verb the question uses: trust the text over the model's claim."""
    lowered = text.lower()
    for verb in sorted(verbs, key=len, reverse=True):
        if lowered.startswith(verb):
            return verb
    for verb in verbs:
        if re.search(rf"\b{re.escape(verb)}\b", lowered):
            return verb
    claimed = claimed.strip().lower()
    return claimed if claimed in verbs else verbs[0]


class NoSourceMaterial(ValueError):
    pass


class QuestionGenerator:
    def __init__(self, subject: LoadedSubject, llm, embedder, settings: Settings, rng: Optional[random.Random] = None):
        self.subject = subject
        self.llm = llm
        self.embedder = embedder
        self.settings = settings
        self.rng = rng or random.Random()
        self._accepted: List[np.ndarray] = []  # vectors of questions already produced
        self._topic_uses: Dict[str, int] = {}

    def resolve_topic(self, query: str):
        """Map free text to (unit number, topic name). Exact syllabus topics keep
        their unit; anything else is treated as a free keyword search."""
        syllabus = self.subject.meta.syllabus
        found = syllabus.find_topic(query) if syllabus else None
        if found:
            return found[0].number, found[1].name
        return 0, query.strip()

    def _unit_title(self, unit: int) -> str:
        syllabus = self.subject.meta.syllabus
        for candidate in syllabus.units if syllabus else []:
            if candidate.number == unit:
                return candidate.title
        return ""

    def _contexts(self, topic: str, unit: int = 0):
        """Choose the passages a question is written from.

        One passage is the anchor: the best match, or the next-best on repeat
        requests so the same topic yields new angles. The rest must come from
        the same part of the book (within a few pages of the anchor), because
        passages from different chapters pull a question in different directions.
        """
        n = self.settings.context_chunks
        retriever = self.subject.retriever
        hits = retriever.search(topic, k=n * 4, context_query=topic_query(topic, self._unit_title(unit)))
        best = max((h.similarity for h in hits), default=0.0)
        if best < self.settings.min_similarity:
            raise NoSourceMaterial(f"The uploaded material does not cover '{topic}'.")
        floor = max(self.settings.min_similarity, self.settings.focus_ratio * hits[0].similarity)
        strong = [h for h in hits if h.similarity >= floor] or hits[:1]

        uses = self._topic_uses.get(topic.lower(), 0)
        self._topic_uses[topic.lower()] = uses + 1
        anchor = strong[uses % len(strong)]

        def nearby(chunk) -> bool:
            return (
                chunk.source == anchor.chunk.source
                and abs(chunk.page_start - anchor.chunk.page_start) <= self.settings.neighbor_pages
            )

        picked = [anchor] + [h for h in strong if h is not anchor and nearby(h.chunk)][: n - 1]
        if len(picked) < n:
            # Fill up with the text that directly follows the anchor in the book.
            taken = {h.chunk.id for h in picked}
            following = [
                i for i in range(anchor.chunk.id + 1, min(anchor.chunk.id + n, len(self.subject.chunks)))
                if i not in taken and nearby(self.subject.chunks[i])
            ][: n - len(picked)]
            picked += retriever.as_hits(topic, following)
        return sorted(picked, key=lambda h: h.chunk.id)  # reading order

    def _is_duplicate(self, vector: np.ndarray) -> bool:
        return any(float(vector @ other) >= self.settings.duplicate_similarity for other in self._accepted)

    def generate(
        self,
        topic: str,
        marks: int,
        unit: int = 0,
        bloom_levels: Optional[List[str]] = None,
        avoid_verb: Optional[str] = None,
    ) -> Question:
        hits = self._contexts(topic, unit)
        unit_title = self._unit_title(unit)
        unit_line = f"\nCourse unit (background only, do not mention it): {unit_title}" if unit_title else ""
        context = "\n\n".join(f"[{i}] {h.chunk.text}" for i, h in enumerate(hits, start=1))
        levels = normalize_levels(bloom_levels, marks)
        started = time.perf_counter()
        problem, last_error = "", None

        for attempt in range(1, self.settings.max_attempts + 1):
            bloom = self.rng.choice(levels)
            verbs = [v for v in BLOOM_VERBS[bloom] if v != avoid_verb] or BLOOM_VERBS[bloom]
            extra = f"- A previous attempt was rejected because it {problem}. Fix that.\n" if problem else ""
            prompt = PROMPT.format(
                topic=topic, unit_line=unit_line, marks=marks, guidance=answer_guidance(marks), bloom=bloom,
                verbs=", ".join(verbs), extra=extra, context=context,
            )
            try:
                reply = self.llm.generate_json(prompt)
            except LLMError as exc:
                last_error = exc
                if "Could not reach" in str(exc) or "does not have the model" in str(exc):
                    raise  # retrying cannot fix a missing server or model
                problem = "was not valid JSON"
                continue
            text = clean_question(reply.get("question", ""))
            problem = validate_question(text, marks) or ""
            if problem:
                continue
            verb = pick_verb(text, str(reply.get("verb", "")), verbs)
            vector = self.embedder.encode([text])[0]
            if self._is_duplicate(vector):
                problem = "repeated an earlier question"
                continue
            chunk_vectors = self.subject.vectors[[h.chunk.id for h in hits]]
            grounding = float(np.max(chunk_vectors @ vector))
            if grounding < self.settings.min_grounding and attempt < self.settings.max_attempts:
                # Far from every source passage: the model probably wrote from its
                # own memory. Try again; on the last attempt, keep it with its low score.
                problem = "was not based closely enough on the source material given"
                continue

            self._accepted.append(vector)
            return Question(
                text=text,
                marks=marks,
                bloom=bloom,
                verb=verb,
                unit=unit,
                topic=topic,
                sources=[
                    SourceRef(
                        chunk_id=h.chunk.id, source=h.chunk.source, page_start=h.chunk.page_start,
                        page_end=h.chunk.page_end, score=round(h.similarity, 4),
                    )
                    for h in hits
                ],
                grounding=round(grounding, 4),
                latency_ms=int((time.perf_counter() - started) * 1000),
                attempts=attempt,
            )
        if last_error and not problem:
            raise last_error
        raise LLMError(
            f"No usable question for '{topic}' after {self.settings.max_attempts} attempts "
            f"(last problem: it {problem})."
        )
