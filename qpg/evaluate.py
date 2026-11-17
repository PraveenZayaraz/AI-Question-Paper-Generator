"""Evaluation harness: generate questions for a sample of topics and measure
quality and speed, so changes to the pipeline can be compared with numbers."""
from __future__ import annotations

import random
import statistics
from typing import List, Optional

from qpg.generate import NoSourceMaterial, QuestionGenerator
from qpg.llm import LLMError


def run_eval(generator: QuestionGenerator, n: int = 20, marks: int = 5, seed: Optional[int] = 0) -> dict:
    syllabus = generator.subject.meta.syllabus
    topics = [(u.number, t.name) for u in syllabus.units for t in u.topics]
    covered = [(u.number, t.name) for u in syllabus.units for t in u.topics if t.support > 0]
    rng = random.Random(seed)
    sample = covered if len(covered) <= n else rng.sample(covered, n)

    rows: List[dict] = []
    failures = 0
    for unit, topic in sample:
        try:
            q = generator.generate(topic, marks, unit=unit)
        except (LLMError, NoSourceMaterial) as exc:
            if "Could not reach" in str(exc):
                raise
            failures += 1
            continue
        rows.append(
            {"unit": unit, "topic": topic, "question": q.text, "bloom": q.bloom, "verb": q.verb,
             "grounding": q.grounding, "latency_ms": q.latency_ms, "attempts": q.attempts,
             "verb_used": q.verb.lower() in q.text.lower()}
        )

    latencies = sorted(r["latency_ms"] for r in rows)
    summary = {
        "llm": getattr(generator.llm, "name", "unknown"),
        "embedder": generator.embedder.name,
        "questions_requested": len(sample),
        "questions_generated": len(rows),
        "failures": failures,
        "syllabus_topics": len(topics),
        "topics_covered_by_material": len(covered),
        # Mean cosine similarity between each question and its best source chunk.
        "mean_grounding": round(statistics.mean(r["grounding"] for r in rows), 4) if rows else None,
        "mean_latency_ms": round(statistics.mean(latencies)) if rows else None,
        "p95_latency_ms": latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if rows else None,
        "first_try_rate": round(sum(r["attempts"] == 1 for r in rows) / len(rows), 3) if rows else None,
        "verb_adherence": round(sum(r["verb_used"] for r in rows) / len(rows), 3) if rows else None,
    }
    return {"summary": summary, "questions": rows}


def to_markdown(summary: dict) -> str:
    lines = ["| Metric | Value |", "|---|---|"]
    lines += [f"| {key.replace('_', ' ')} | {value} |" for key, value in summary.items()]
    return "\n".join(lines)
