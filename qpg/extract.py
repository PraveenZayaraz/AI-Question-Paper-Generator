"""Turn PDF, Word and text files into a list of (page number, text) pairs."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple

Page = Tuple[int, str]
SUPPORTED = {".pdf", ".docx", ".txt", ".md"}


def extract_pages(path) -> List[Page]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages = _pdf_pages(path)
    elif suffix == ".docx":
        pages = [(1, _docx_text(path))]
    elif suffix in {".txt", ".md"}:
        pages = [(1, path.read_text(encoding="utf-8", errors="replace"))]
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Use one of: {', '.join(sorted(SUPPORTED))}")
    return [(n, t) for n, t in pages if t and t.strip()]


def extract_text(path) -> str:
    return "\n".join(text for _, text in extract_pages(path))


def _pdf_pages(path: Path) -> List[Page]:
    try:
        import fitz  # PyMuPDF: fastest and most accurate of the three

        with fitz.open(str(path)) as doc:
            return [(i + 1, page.get_text("text")) for i, page in enumerate(doc)]
    except ImportError:
        pass
    try:
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            return [(i + 1, page.extract_text() or "") for i, page in enumerate(pdf.pages)]
    except ImportError:
        pass
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install PyMuPDF, pdfplumber or pypdf to read PDF files.") from exc
    reader = PdfReader(str(path))
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def _docx_text(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    lines = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            cells = []
            for cell in row.cells:
                text = cell.text.strip()
                if text and text not in cells:  # merged cells repeat their text
                    cells.append(text)
            lines.append(" : ".join(cells))
    return "\n".join(lines)


def _is_running_header(line: str) -> bool:
    """Textbook headers such as "SEC. 1.4 REFERENCE MODELS 43" or "44 INTRODUCTION CHAP. 1":
    short, upper-case, and carrying a page or section number at one end."""
    line = line.strip()
    letters = [c for c in line if c.isalpha()]
    if not (4 <= len(letters) and len(line) <= 70):
        return False
    if sum(c.isupper() for c in letters) / len(letters) < 0.85:
        return False
    return bool(re.match(r"^\d+\b", line) or re.search(r"\b\d+$", line))


def strip_repeated_lines(pages: List[Page], edge_lines: int = 2, min_share: float = 0.4) -> List[Page]:
    """Remove running headers and footers.

    A line that shows up at the top or bottom of at least `min_share` of the
    pages (ignoring digits, so "Page 12" matches "Page 13") is page furniture,
    not content.
    """
    if len(pages) < 5:
        return pages

    def key(line: str) -> str:
        return re.sub(r"\d+", "#", line.strip().lower())

    counts: Counter = Counter()
    for _, text in pages:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        n = min(edge_lines, len(lines) // 3)  # never treat most of a short page as its edges
        edges = lines[:n] + lines[-n:] if n else []
        counts.update({key(ln) for ln in edges})
    limit = max(3, int(len(pages) * min_share))
    repeated = {k for k, c in counts.items() if c >= limit}

    cleaned = []
    for number, text in pages:
        lines = text.splitlines()
        kept = []
        content_idx = [i for i, ln in enumerate(lines) if ln.strip()]
        n = min(edge_lines, len(content_idx) // 3)
        edge_idx = set(content_idx[:n] + content_idx[-n:]) if n else set()
        first_last = set(content_idx[:1] + content_idx[-1:]) if len(content_idx) > 3 else set()
        for i, ln in enumerate(lines):
            if i in edge_idx and key(ln) in repeated:
                continue
            if i in first_last and _is_running_header(ln):
                continue
            kept.append(ln)
        cleaned.append((number, "\n".join(kept)))
    return cleaned
