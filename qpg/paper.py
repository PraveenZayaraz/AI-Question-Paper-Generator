"""Assemble a full question paper from a blueprint and export it to Word."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, List, Optional, Set

from qpg.generate import NoSourceMaterial, QuestionGenerator
from qpg.models import Blueprint, Paper, PaperItem, PaperSection, Question, SectionSpec, Syllabus, Unit


def default_blueprint(n_units: int) -> Blueprint:
    n = max(1, n_units)
    return Blueprint(
        title="Question Paper",
        sections=[
            SectionSpec(name="Part A", marks=2, count=n, bloom=["Remember", "Understand"]),
            SectionSpec(name="Part B", marks=8, count=n, either_or=True, split=True,
                        bloom=["Understand", "Apply", "Analyze"]),
            SectionSpec(name="Part C", marks=10, count=1, either_or=True, bloom=["Evaluate", "Create"]),
        ],
    )


def count_llm_tasks(blueprint: Blueprint) -> int:
    return sum(s.count * (2 if s.either_or else 1) for s in blueprint.sections)


class TopicPicker:
    """Chooses topics so a paper spreads across the syllabus: unused topics
    first, and topics with more textbook support are more likely."""

    def __init__(self, syllabus: Syllabus, rng: random.Random):
        self.syllabus = syllabus
        self.rng = rng
        self.used: Set[str] = set()

    def pick(self, unit: Unit, exclude: Set[str] = frozenset()) -> Optional[str]:
        covered = [t for t in unit.topics if t.support > 0 and t.name not in exclude]
        if not covered:
            return None
        fresh = [t for t in covered if t.name not in self.used] or covered
        choice = self.rng.choices(fresh, weights=[t.support for t in fresh], k=1)[0]
        self.used.add(choice.name)
        return choice.name


def _split_marks(marks: int, rng: random.Random) -> List[int]:
    if marks < 6:
        return [marks]
    return rng.choice([[marks], [marks - marks // 2, marks // 2], [marks - 2, 2]])


def build_paper(
    paper: Paper,
    syllabus: Syllabus,
    generator: QuestionGenerator,
    rng: Optional[random.Random] = None,
    on_progress: Optional[Callable[[Paper], None]] = None,
) -> Paper:
    rng = rng or generator.rng
    units = [u for u in syllabus.units if any(t.support > 0 for t in u.topics)]
    if not units:
        raise ValueError("None of the syllabus topics are covered by the uploaded material.")
    picker = TopicPicker(syllabus, rng)
    paper.total = count_llm_tasks(paper.blueprint)
    number, turn = 0, 0

    def make_option(section: SectionSpec, unit: Unit, exclude: Set[str]) -> Optional[List[Question]]:
        # Try a few topics: one may turn out to have no usable material.
        for _ in range(4):
            topic = picker.pick(unit, exclude)
            if topic is None:
                return None
            parts = _split_marks(section.marks, rng) if section.split else [section.marks]
            try:
                questions: List[Question] = []
                for marks in parts:
                    avoid = questions[-1].verb if questions else None
                    # A small sub-part tests recall, whatever level the section asks for.
                    levels = section.bloom if marks > 2 or len(parts) == 1 else None
                    questions.append(
                        generator.generate(topic, marks, unit=unit.number, bloom_levels=levels, avoid_verb=avoid)
                    )
                exclude.add(topic)
                return questions
            except NoSourceMaterial:
                exclude.add(topic)
        return None

    for section in paper.blueprint.sections:
        out = PaperSection(name=section.name, marks=section.marks)
        paper.sections.append(out)
        for _ in range(section.count):
            unit = units[turn % len(units)]  # round-robin so every unit is examined
            turn += 1
            exclude: Set[str] = set()
            options = []
            for _ in range(2 if section.either_or else 1):
                option = make_option(section, unit, exclude)
                if option:
                    options.append(option)
                paper.progress += 1
                if on_progress:
                    on_progress(paper)
            if options:
                number += 1
                out.items.append(PaperItem(number=number, options=options))
    if number == 0:
        raise ValueError("No questions could be generated for this paper.")
    paper.status = "ready"
    return paper


def _page_list(question: Question) -> str:
    """Compact source list: "book.pdf p.327-328, 341" instead of repeating the file name."""
    by_source: dict = {}
    for ref in question.sources:
        span = str(ref.page_start) if ref.page_end == ref.page_start else f"{ref.page_start}-{ref.page_end}"
        pages = by_source.setdefault(ref.source, [])
        if span not in pages:
            pages.append(span)
    return "; ".join(f"{source} p.{', '.join(pages)}" for source, pages in by_source.items())


def export_docx(paper: Paper, path: Path) -> Path:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt

    doc = Document()
    doc.styles["Normal"].font.size = Pt(11)

    heading = doc.add_heading(paper.blueprint.title, level=0)
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    info = doc.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    info.add_run(paper.subject_name).bold = True
    facts = [f"Maximum marks: {paper.blueprint.total_marks}"]
    if paper.blueprint.duration_minutes:
        facts.insert(0, f"Time: {paper.blueprint.duration_minutes} minutes")
    doc.add_paragraph("    ".join(facts)).alignment = WD_ALIGN_PARAGRAPH.CENTER

    def add_question(label: str, question: Question, indent: float) -> None:
        # Hanging indent: the label sits in the margin and wrapped lines align with the text.
        para = doc.add_paragraph()
        fmt = para.paragraph_format
        fmt.left_indent = Inches(indent)
        fmt.first_line_indent = Inches(-indent)
        fmt.tab_stops.add_tab_stop(Inches(indent))
        fmt.space_after = Pt(6)
        para.add_run(f"{label}\t{question.text}")
        para.add_run(f"  [{question.marks}]").bold = True

    letters = "ab"
    numerals = ["i", "ii", "iii", "iv"]
    for spec, section in zip(paper.blueprint.sections, paper.sections):
        doc.add_heading(f"{section.name}  ({len(section.items)} × {section.marks} = "
                        f"{len(section.items) * section.marks} marks)", level=1)
        choice = any(len(item.options) > 1 for item in section.items)
        note = "Answer either (a) or (b) of each question." if choice else "Answer all questions."
        doc.add_paragraph().add_run(note).italic = True
        nested = any(len(option) > 1 for item in section.items for option in item.options)
        indent = 0.4 + (0.3 if choice else 0) + (0.3 if nested else 0)
        for item in section.items:
            for o, option in enumerate(item.options):
                for q, question in enumerate(option):
                    parts = []
                    if o == 0 and q == 0:
                        parts.append(f"{item.number}.")
                    if len(item.options) > 1 and q == 0:
                        parts.append(f"({letters[o]})")
                    if len(option) > 1:
                        parts.append(f"({numerals[min(q, 3)]})")
                    label = " ".join(parts)
                    if o > 0 and q == 0:
                        label = f"\u2002\u2002{label}"  # line (b) up under (a)
                    if len(option) > 1 and q > 0:
                        label = f"\u2003\u2003{label}"  # line the sub-part up under (i)
                    add_question(label, question, indent)
                if o == 0 and len(item.options) > 1:
                    doc.add_paragraph("(OR)").alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Examiner's page: where every question came from, so it can be checked.
    doc.add_page_break()
    doc.add_heading("Examiner's key: sources and cognitive levels", level=1)
    table = doc.add_table(rows=1, cols=5)
    table.style = "Light Grid Accent 1"
    table.autofit = False
    widths = [Inches(0.7), Inches(2.0), Inches(1.0), Inches(0.6), Inches(1.9)]
    for cell, title in zip(table.rows[0].cells, ["Q", "Topic", "Bloom level", "Marks", "Source pages"]):
        cell.text = title
    for section in paper.sections:
        for item in section.items:
            for o, option in enumerate(item.options):
                for q, question in enumerate(option):
                    row = table.add_row().cells
                    label = f"{item.number}{letters[o] if len(item.options) > 1 else ''}"
                    row[0].text = label + (f"({numerals[min(q, 3)]})" if len(option) > 1 else "")
                    row[1].text = question.topic
                    row[2].text = question.bloom
                    row[3].text = str(question.marks)
                    row[4].text = _page_list(question)
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = width
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    return path
