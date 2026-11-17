"""Command line interface.

    python -m qpg ingest --name "Computer Networks" --syllabus syllabus.pdf --textbook book.pdf
    python -m qpg list
    python -m qpg ask <subject-id> --unit 2 --marks 8 --count 3
    python -m qpg paper <subject-id>
    python -m qpg eval <subject-id> --n 20
    python -m qpg serve
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qpg.evaluate import run_eval, to_markdown
from qpg.service import Service
from qpg.store import now_iso


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="qpg", description="Syllabus-aligned question paper generator")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="index a syllabus and its textbooks")
    p.add_argument("--name", required=True)
    p.add_argument("--syllabus", required=True, type=Path)
    p.add_argument("--textbook", required=True, type=Path, nargs="+")

    sub.add_parser("list", help="list subjects")

    p = sub.add_parser("ask", help="generate questions")
    p.add_argument("subject")
    p.add_argument("--unit", type=int)
    p.add_argument("--topic")
    p.add_argument("--marks", type=int, default=5)
    p.add_argument("--count", type=int, default=1)

    p = sub.add_parser("paper", help="generate a full paper as a Word file")
    p.add_argument("subject")
    p.add_argument("--blueprint", type=Path, help="JSON file describing the sections")

    p = sub.add_parser("eval", help="measure question quality and speed")
    p.add_argument("subject")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--marks", type=int, default=5)
    p.add_argument("--out", type=Path, default=Path("results"))

    p = sub.add_parser("serve", help="start the web API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    if args.command == "serve":
        import uvicorn

        uvicorn.run("qpg.api:app", host=args.host, port=args.port)
        return 0

    svc = Service()
    try:
        if args.command == "ingest":
            meta = svc.store.create_subject(args.name)
            meta = svc.ingest(meta.id, args.textbook, syllabus_path=args.syllabus, progress=print)
            topics = [t for u in meta.syllabus.units for t in u.topics]
            covered = sum(t.support > 0 for t in topics)
            print(f"\nReady: {meta.id}")
            print(f"{meta.n_chunks} chunks, {len(meta.syllabus.units)} units, "
                  f"{covered} of {len(topics)} topics covered by the material "
                  f"(syllabus read by: {meta.syllabus.parser})")
        elif args.command == "list":
            for meta in svc.store.list_subjects():
                print(f"{meta.id:48} {meta.status:10} {meta.name}")
        elif args.command == "ask":
            for i, q in enumerate(
                svc.generate_questions(args.subject, args.marks, args.count, unit=args.unit, topic=args.topic), 1
            ):
                pages = ", ".join(f"p.{s.page_start}" for s in q.sources)
                print(f"{i}. [{q.marks}M, {q.bloom}] {q.text}\n   topic: {q.topic} | sources: {pages} | "
                      f"grounding {q.grounding} | {q.latency_ms} ms")
        elif args.command == "paper":
            from qpg.models import Blueprint

            blueprint = Blueprint.model_validate_json(args.blueprint.read_text("utf-8")) if args.blueprint else None
            paper = svc.run_paper(svc.create_paper(args.subject, blueprint))
            print(f"Saved: {svc.store.paper_docx(paper.id)}")
        elif args.command == "eval":
            result = run_eval(svc.generator(args.subject, seed=0), n=args.n, marks=args.marks)
            args.out.mkdir(parents=True, exist_ok=True)
            out = args.out / f"eval-{now_iso().replace(':', '').replace('+0000', 'Z')}.json"
            out.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(to_markdown(result["summary"]))
            print(f"\nSaved: {out}")
    except (KeyError, ValueError, RuntimeError) as exc:
        print(f"Error: {str(exc).strip(chr(39))}", file=sys.stderr)
        return 1
    return 0
