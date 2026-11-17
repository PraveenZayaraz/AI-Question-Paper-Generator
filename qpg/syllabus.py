"""Read a syllabus in any common layout and turn it into units and topics.

Strategy: a rule-based parser runs first because it is instant and cannot
invent topics. If it finds too little structure, the language model is asked
to do the extraction, and every topic it returns is checked against the
original text so made-up topics are dropped.
"""
from __future__ import annotations

import re
from typing import List, Optional

from qpg.models import Syllabus, Topic, Unit

_UNIT_HEADER = re.compile(
    r"^\s*(unit|module|chapter|part|week|section)\s*[-–—:.#]?\s*([ivxlcdm]+|\d{1,2})\b\s*[-–—:.)]*\s*(.*)$",
    re.I,
)
_STOP_SECTION = re.compile(
    r"^\s*(text\s*books?|reference\s*books?|references?|course\s+outcomes?|learning\s+outcomes?|"
    r"suggested\s+readings?|web\s*(sites?|resources|links)|e-?resources|"
    r"(total|lecture|tutorial|practical)\s*(hours|periods)?\s*:)",
    re.I,
)
_HOURS = re.compile(
    r"\(?\b\d{1,2}\s*(hours?|hrs?|periods?|lectures?|h)\b\.?\)?"  # "9 Hours", "(9 hrs)"
    r"|\(?\b(hours?|hrs?|periods?|lectures?)\s*[:=-]?\s*\d{1,2}\b\)?"  # "Periods: 9"
    r"|\(\s*\d{1,2}\s*\)\s*$",
    re.I,
)
# En and em dashes always separate topics. A hyphen only does when it has a
# space beside it, so "Collision-Free" and "Traffic-Aware" stay whole.
_TOPIC_SPLIT = re.compile(r"\s*[–—]\s*|\s+-\s*|\s*-\s+|[;,\n•·●▪■◦]|\.\s+|:\s+")
_INLINE_SEPARATOR = re.compile(r"[–—;,]|\s-\s")
# Syllabus filler that names no subject matter. "Introduction" under any unit
# retrieves whichever chapter opening happens to match, so it is not a topic.
_NOT_A_TOPIC = frozenset(
    "introduction overview basics example examples summary uses definition definitions concepts "
    "fundamentals preliminaries review revision conclusion".split()
)
_OUTCOME_TAG = re.compile(r"^(CO|PO|LO|CLO|PSO)\s*-?\d+$", re.I)
_WORD = re.compile(r"[a-z0-9]+")
_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def roman_to_int(text: str) -> int:
    total, prev = 0, 0
    for char in reversed(text.lower()):
        value = _ROMAN[char]
        total += -value if value < prev else value
        prev = max(prev, value)
    return total


def _clean_topic(raw: str) -> str:
    topic = _HOURS.sub(" ", raw)
    # Strip list numbering ("1.2 ", "a) ") but keep names such as "802.11 MAC".
    topic = re.sub(r"^\s*(\d{1,2}(\.\d{1,2})*[.)]?|[a-z][.)])\s+", "", topic, flags=re.I)
    topic = re.sub(r"^\s*(and|&)\s+", "", topic, flags=re.I)
    topic = re.sub(r"\s+", " ", topic).strip(" .:-–—\t")
    return topic


def unwrap_lines(body: str) -> str:
    """Decide whether line breaks separate topics or are just word wrap.

    If the text already uses dashes, commas or semicolons at least as often as
    it breaks lines, it is running prose that wrapped, and a line break in the
    middle of "Theoretical Basis For / Communication" must not split the topic.
    """
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if len(lines) < 2 or len(_INLINE_SEPARATOR.findall(body)) < len(lines):
        return body
    kept = [ln for ln in lines if not _OUTCOME_TAG.match(ln)]
    return re.sub(r"(\w)-\s+(?=[a-z])", r"\1-", " ".join(kept))


def split_topics(body: str) -> List[str]:
    body = unwrap_lines(body)
    topics, seen = [], set()
    for piece in _TOPIC_SPLIT.split(body):
        topic = _clean_topic(piece)
        key = topic.lower()
        if _OUTCOME_TAG.match(topic) or key in _NOT_A_TOPIC:
            continue
        if 3 <= len(topic) <= 90 and re.search(r"[a-zA-Z]{2}", topic) and key not in seen:
            seen.add(key)
            topics.append(topic)
    return topics


def parse_heuristic(text: str) -> Syllabus:
    units: List[Unit] = []
    bodies: List[List[str]] = []
    collecting = False
    for line in text.splitlines():
        if not line.strip():
            continue
        header = _UNIT_HEADER.match(line)
        if header:
            numeral, rest = header.group(2), _HOURS.sub(" ", header.group(3)).strip()
            number = int(numeral) if numeral.isdigit() else roman_to_int(numeral)
            # "Unit 1: Introduction - Uses of networks, ..." -> title + inline topics
            parts = re.split(r"\s*[–—]\s*|\s-\s|:\s+", rest, maxsplit=1)
            title = _clean_topic(parts[0]) if parts[0] else ""
            inline = parts[1] if len(parts) > 1 else ""
            if len(parts) == 1 and len(_TOPIC_SPLIT.split(rest)) > 2:
                title, inline = "", rest  # the header line is itself a topic list
            units.append(Unit(number=number, title=title or f"{header.group(1).title()} {number}"))
            bodies.append([inline] if inline else [])
            collecting = True
        elif _STOP_SECTION.match(line):
            collecting = False
        elif collecting:
            bodies[-1].append(line)

    for unit, body in zip(units, bodies):
        unit.topics = [Topic(name=t) for t in split_topics("\n".join(body))]
    units = [u for u in units if u.topics]

    if not units:  # no unit headers at all: treat the whole document as one unit
        topics = [Topic(name=t) for t in split_topics(text)]
        if topics:
            units = [Unit(number=1, title="Syllabus", topics=topics)]
    return Syllabus(units=units, parser="heuristic")


def topic_query(topic: str, unit_title: str = "") -> str:
    """Search text for a topic. Short names such as "Design Issues" mean little
    alone, so the unit title is added for context."""
    if unit_title and unit_title.lower() not in topic.lower() and not unit_title.lower().startswith(("unit", "module")):
        return f"{topic} ({unit_title})"
    return topic


def looks_structured(syllabus: Syllabus) -> bool:
    if len(syllabus.units) < 2:
        return False
    return sum(len(u.topics) for u in syllabus.units) / len(syllabus.units) >= 3


_PROMPT = """You extract the structure of a university course syllabus.

Return JSON in exactly this shape:
{{"units": [{{"title": "unit title", "topics": ["topic 1", "topic 2"]}}]}}

Rules:
- Use only units and topics that appear in the syllabus text. Do not add or rename topics.
- A topic is a short noun phrase (2 to 8 words).
- Skip textbooks, references, course outcomes, hours and marks.

Syllabus:
\"\"\"
{text}
\"\"\"
"""


def _grounded(topic: str, source_words: set) -> bool:
    words = [w for w in _WORD.findall(topic.lower()) if len(w) > 2]
    if not words:
        return False
    return sum(w in source_words for w in words) / len(words) >= 0.6


def parse_with_llm(text: str, llm) -> Optional[Syllabus]:
    try:
        data = llm.generate_json(_PROMPT.format(text=text[:8000]), temperature=0.0)
    except Exception:
        return None
    source_words = set(_WORD.findall(text.lower()))
    units: List[Unit] = []
    for raw in data.get("units") or []:
        if not isinstance(raw, dict):
            continue
        seen, topics = set(), []
        for name in raw.get("topics") or []:
            name = _clean_topic(str(name))
            if 3 <= len(name) <= 90 and name.lower() not in seen and _grounded(name, source_words):
                seen.add(name.lower())
                topics.append(Topic(name=name))
        if topics:
            number = len(units) + 1
            title = _clean_topic(str(raw.get("title") or "")) or f"Unit {number}"
            units.append(Unit(number=number, title=title, topics=topics))
    return Syllabus(units=units, parser="llm") if units else None


def parse_syllabus(text: str, llm=None) -> Syllabus:
    heuristic = parse_heuristic(text)
    if looks_structured(heuristic) or llm is None:
        return heuristic
    from_llm = parse_with_llm(text, llm)
    if from_llm is None:
        return heuristic
    n_llm = sum(len(u.topics) for u in from_llm.units)
    n_rules = sum(len(u.topics) for u in heuristic.units)
    # Prefer the model when it recovered unit structure the rules missed.
    if len(from_llm.units) > len(heuristic.units) or n_llm >= n_rules:
        return from_llm
    return heuristic
