import pytest

from qpg.chunking import chunk_pages, clean_text, split_sentences
from qpg.extract import extract_pages, strip_repeated_lines

SENTENCE = "The quick brown fox jumps over the lazy dog near the quiet river bank today."


def test_clean_text_rejoins_hyphenated_words():
    assert clean_text("trans-\nmission   media\n\nare cables") == "transmission media are cables"


def test_long_text_without_sentence_breaks_is_still_split():
    parts = split_sentences("word " * 400, max_chars=200)
    assert all(len(p) <= 200 for p in parts) and len(parts) > 5


def test_chunks_respect_size_overlap_and_pages():
    second = "Second page sentence about routers and switches in a network."
    pages = [(1, " ".join([SENTENCE] * 6)), (2, " ".join([second] * 8))]
    chunks = chunk_pages(pages, "book.pdf", max_chars=300, min_words=10)
    assert len(chunks) > 2
    assert [c.id for c in chunks] == list(range(len(chunks)))
    assert all(len(c.text) <= 300 + len(SENTENCE) for c in chunks)
    assert chunks[0].page_start == 1 and chunks[-1].page_end == 2
    # the last sentence of one chunk opens the next
    assert chunks[1].text.startswith(chunks[0].text.split(". ")[-1][:20])


def test_short_fragments_are_dropped():
    assert chunk_pages([(1, "Too short to be useful.")], "x.txt") == []


def test_repeated_headers_and_footers_are_removed():
    pages = [(i, f"Networks 101 - Chapter 3\nReal content number {i} lives here.\nPage {i}") for i in range(1, 9)]
    cleaned = strip_repeated_lines(pages)
    assert all("Networks 101" not in text and "Page" not in text for _, text in cleaned)
    assert all("Real content" in text for _, text in cleaned)


def test_extract_txt_and_docx(tmp_path):
    txt = tmp_path / "a.txt"
    txt.write_text("hello world", encoding="utf-8")
    assert extract_pages(txt) == [(1, "hello world")]

    from docx import Document

    doc = Document()
    doc.add_paragraph("Unit 1 Basics")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text, table.rows[0].cells[1].text = "Unit 2", "Trees, Heaps"
    path = tmp_path / "s.docx"
    doc.save(str(path))
    text = extract_pages(path)[0][1]
    assert "Unit 1 Basics" in text and "Unit 2 : Trees, Heaps" in text


def test_extract_pdf_keeps_page_numbers(tmp_path):
    canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    path = tmp_path / "b.pdf"
    c = canvas.Canvas(str(path))
    c.drawString(72, 720, "First page about routing")
    c.showPage()
    c.drawString(72, 720, "Second page about congestion")
    c.save()
    pages = extract_pages(path)
    assert [n for n, _ in pages] == [1, 2]
    assert "congestion" in pages[1][1]


def test_unsupported_file_type(tmp_path):
    bad = tmp_path / "x.exe"
    bad.write_bytes(b"0")
    with pytest.raises(ValueError):
        extract_pages(bad)


def test_numbered_running_headers_are_removed():
    body = "Layers hide detail from each other.\nEach layer offers a service.\nProtocols define the rules."
    endings = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    pages = [
        (i, f"SEC. 1.{i} REFERENCE MODELS {40 + i}\n{body}\nThe OSI model has 7 layers, said {endings[i - 1]}.")
        for i in range(1, 7)
    ]
    cleaned = strip_repeated_lines(pages)
    assert all("REFERENCE MODELS" not in text for _, text in cleaned)
    assert all("has 7 layers" in text and "Layers hide" in text for _, text in cleaned)


def test_exercise_lists_and_bibliographies_are_not_indexed():
    from qpg.chunking import is_back_matter

    problems = ("11. A router has 20 bits for the network part. How many hosts fit? 12. Compute the checksum for the "
                "frame below using the generator given. 13. Explain why a class B network wastes address space today.")
    references = ("SMITH, J. et al.: Routing at scale, Proc. Conf., pp. 12-19, 2001. JONES, K. et al.: Congestion, "
                  "Journal of Nets, vol. 4, pp. 100-120, 1999.")
    prose = "Version 4 of IP uses 32 bit addresses. Version 6 uses 128 bit addresses, written in hexadecimal groups."
    assert is_back_matter(problems) and is_back_matter(references) and not is_back_matter(prose)
    assert chunk_pages([(1, problems + " " + problems)], "book.pdf", min_words=10) == []
