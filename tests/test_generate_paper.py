import pytest

from qpg.bloom import BLOOM_VERBS, levels_for_marks, normalize_levels
from qpg.generate import NoSourceMaterial, clean_question, validate_question
from qpg.llm import FakeLLM, LLMError, parse_json_object
from qpg.models import Blueprint, SectionSpec
from qpg.paper import count_llm_tasks, default_blueprint
from qpg.store import Store
from tests.conftest import EXAMPLES


def test_ingest_builds_a_ready_subject(service, subject_id):
    meta = service.store.get_meta(subject_id)
    assert meta.status == "ready" and meta.n_chunks >= 12
    assert [u.number for u in meta.syllabus.units] == [1, 2, 3]
    topics = [t for u in meta.syllabus.units for t in u.topics]
    assert len(topics) == 12
    assert all(t.support > 0 for t in topics), [t.name for t in topics if t.support == 0]


def test_ingest_records_failure_reason(service, tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("too short", encoding="utf-8")
    meta = service.store.create_subject("Broken")
    with pytest.raises(ValueError):
        service.ingest(meta.id, [empty], syllabus_text="Unit 1 A\nAlpha, Beta")
    failed = service.store.get_meta(meta.id)
    assert failed.status == "failed" and "No readable text" in failed.message


def test_bloom_levels_scale_with_marks():
    assert levels_for_marks(2) == ["Remember", "Understand"]
    assert "Evaluate" in levels_for_marks(10)
    assert normalize_levels(["analyze", "nonsense"], 2) == ["Analyze"]
    assert normalize_levels([], 2) == levels_for_marks(2)


def test_question_cleaning_and_validation():
    assert clean_question('  "Here is Explain framing."  ') == "Explain framing."
    assert clean_question("(8 marks) Compare go back N with ARQ.") == "Compare go back N with ARQ."
    assert validate_question("Explain how bit stuffing works.") is None
    assert validate_question("Why?") == "was too short"
    assert "source material" in validate_question("According to the passage, what is framing used for?")
    assert "source material" in validate_question(
        "Implement a switch, considering the limitations discussed in the course material.")
    assert "source material" in validate_question("Name the first carrier sense protocol studied in this unit.")
    assert "source material" in validate_question("Compare the two agents, as described in [1].")
    assert validate_question("Define CRC. Answer: a checksum method.") == "included an answer"
    two_parts = "Examine the transport layer. How does it serve the layers above?"
    assert "more than one thing" in validate_question(two_parts, marks=2)
    assert validate_question(two_parts, marks=8) is None
    assert "longer than" in validate_question("Explain " + "very " * 60 + "carefully.")
    # ordinary exam wording that must not be mistaken for a reference to the prompt
    assert validate_question("Explain how a router uses the source address in the context of forwarding.") is None
    assert validate_question("Describe the content delivery network architecture and its benefits.") is None


def test_verb_is_read_from_the_question_itself():
    from qpg.generate import pick_verb

    verbs = ["apply", "demonstrate", "solve", "implement", "show how"]
    assert pick_verb("Show how CSMA/CD recovers from a collision.", "apply", verbs) == "show how"
    assert pick_verb("Using a diagram, solve the following.", "", verbs) == "solve"
    assert pick_verb("What happens next?", "Implement", verbs) == "implement"
    assert pick_verb("What happens next?", "explain", verbs) == "apply"


def test_json_parsing_tolerates_code_fences():
    assert parse_json_object('```json\n{"question": "Q?"}\n```') == {"question": "Q?"}
    with pytest.raises(LLMError):
        parse_json_object("no json here")


def test_generated_question_is_grounded_in_the_right_source(service, subject_id):
    gen = service.generator(subject_id, seed=1)
    q = gen.generate("Sliding window protocols", marks=8, unit=2)
    assert q.marks == 8 and q.unit == 2 and q.bloom in ("Apply", "Analyze")
    assert q.verb in BLOOM_VERBS[q.bloom]
    assert q.sources and q.attempts == 1 and 0 < q.grounding <= 1.0
    texts = [gen.subject.chunks[s.chunk_id].text.lower() for s in q.sources]
    assert any("sliding window" in text for text in texts)


def test_sources_for_one_question_come_from_the_same_part_of_the_book(service, settings, tmp_path):
    """Two chapters both mention switches; a question must not mix them."""
    filler = "Unrelated filler sentence about gardening tools and weather patterns for padding. " * 4
    chapter_a = ("A learning bridge or switch builds a table of which port leads to which station. "
                 "Switches forward frames only to the port where the destination lives. ") * 4
    chapter_b = ("Packet switches in the telephone network of the past used circuit switching. "
                 "Those switches reserved a path for the whole call. ") * 4
    from docx import Document

    doc = Document()
    for i in range(40):
        doc.add_paragraph(chapter_a if i == 5 else chapter_b if i == 30 else filler + f" Marker {i}.")
    book = tmp_path / "book.docx"
    doc.save(str(book))

    # Give every paragraph its own "page" so page distance means something.
    import qpg.ingest as ingest_module
    from qpg.extract import extract_pages as real_extract

    def paged(path):
        pages = real_extract(path)
        if str(path).endswith("book.docx"):
            return [(i + 1, line) for i, line in enumerate(pages[0][1].split("\n")) if line.strip()]
        return pages

    original = ingest_module.extract_pages
    ingest_module.extract_pages = paged
    try:
        meta = service.store.create_subject("Switching")
        service.ingest(meta.id, [book], syllabus_text="Unit 1 Devices\nSwitches, Bridges, Hubs\nUnit 2 Other\nA, B, C")
    finally:
        ingest_module.extract_pages = original

    distinct = iter([
        "Explain how a learning switch fills its forwarding table.",
        "Compare circuit switching with frame forwarding in a local network.",
        "Describe what happens when a destination port is unknown.",
    ])
    service.llm = FakeLLM(lambda prompt: {"question": next(distinct)})
    gen = service.generator(meta.id, seed=1)
    for _ in range(3):
        q = gen.generate("Switches", marks=8, unit=1)
        pages = [s.page_start for s in q.sources]
        assert max(pages) - min(pages) <= settings.neighbor_pages, pages


def test_bad_output_is_retried_then_accepted(service, subject_id):
    replies = iter([
        {"question": "According to the passage, what is framing?"},
        {"wrong_key": "x"},
        {"question": "Explain why bit stuffing keeps the flag pattern out of the data."},
    ])
    service.llm = FakeLLM(lambda prompt: next(replies))
    q = service.generator(subject_id, seed=1).generate("Framing", marks=5)
    assert q.attempts == 3 and q.text.startswith("Explain why bit stuffing")


def test_duplicate_questions_are_rejected(service, subject_id):
    service.llm = FakeLLM(lambda prompt: {"question": "Explain how a cyclic redundancy check detects errors."})
    gen = service.generator(subject_id, seed=1)
    gen.generate("Error detection with CRC", marks=5)
    with pytest.raises(LLMError, match="repeated an earlier question"):
        gen.generate("Error detection with CRC", marks=5)


def test_topic_outside_the_material_is_refused(service, subject_id):
    gen = service.generator(subject_id, seed=1)
    with pytest.raises(NoSourceMaterial):
        gen.generate("zebrafish embryology xylophone", marks=5)
    assert service.llm.calls == 0  # refused before spending a model call


def test_free_keyword_search_works_without_an_exact_topic_name(service, subject_id):
    questions = service.generate_questions(subject_id, marks=2, count=1, topic="three way handshake")
    assert questions[0].unit == 0 and questions[0].topic == "three way handshake"


def test_unit_requests_stay_inside_the_unit(service, subject_id):
    questions = service.generate_questions(subject_id, marks=2, count=4, unit=3, seed=3)
    assert len(questions) == 4 and {q.unit for q in questions} == {3}
    assert len({q.topic for q in questions}) == 4  # four different topics, none repeated
    with pytest.raises(ValueError):
        service.generate_questions(subject_id, marks=2, count=1, unit=99)


def test_default_paper_follows_the_blueprint(service, subject_id):
    paper = service.run_paper(service.create_paper(subject_id), seed=5)
    assert paper.status == "ready" and paper.progress == paper.total == count_llm_tasks(paper.blueprint)
    part_a, part_b, part_c = paper.sections
    assert [len(s.items) for s in paper.sections] == [3, 3, 1]
    assert all(len(item.options) == 1 for item in part_a.items)
    assert all(len(item.options) == 2 for item in part_b.items + part_c.items)
    for item in part_b.items:
        for option in item.options:
            assert sum(q.marks for q in option) == 8
        assert item.options[0][0].topic != item.options[1][0].topic
    assert [i.number for s in paper.sections for i in s.items] == list(range(1, 8))
    assert {i.options[0][0].unit for i in part_a.items} == {1, 2, 3}
    assert paper.blueprint.total_marks == 3 * 2 + 3 * 8 + 10

    from docx import Document

    doc = Document(str(service.store.paper_docx(paper.id)))
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "Part B" in text and "(OR)" in text and "Maximum marks: 40" in text
    assert len(doc.tables[0].rows) > 8  # examiner's key lists every question


def test_custom_blueprint(service, subject_id):
    blueprint = Blueprint(title="Quiz 1", duration_minutes=30,
                          sections=[SectionSpec(name="Section 1", marks=5, count=4, bloom=["Apply"])])
    paper = service.run_paper(service.create_paper(subject_id, blueprint), seed=2)
    assert len(paper.sections[0].items) == 4
    assert all(i.options[0][0].bloom == "Apply" for i in paper.sections[0].items)


def test_default_blueprint_scales_with_units():
    assert [s.count for s in default_blueprint(5).sections] == [5, 5, 1]


def test_ids_cannot_escape_the_data_folder(tmp_path):
    store = Store(tmp_path / "subjects", tmp_path / "papers")
    for bad in ["../secrets", "..", "a/b", "C:\\Windows", "", "paper-1/../../x"]:
        with pytest.raises(KeyError):
            store.get_paper(bad)
        with pytest.raises(KeyError):
            store.get_meta(bad)


def test_embedder_mismatch_is_reported(service, subject_id):
    class Other:
        name = "st:other-model"

    with pytest.raises(ValueError, match="Re-ingest"):
        Store(service.settings.subjects_dir, service.settings.papers_dir).load_subject(subject_id, Other())


def test_evaluation_summary(service, subject_id):
    from qpg.evaluate import run_eval, to_markdown

    result = run_eval(service.generator(subject_id, seed=0), n=6, marks=5)
    s = result["summary"]
    assert s["questions_generated"] == 6 and s["failures"] == 0
    assert s["topics_covered_by_material"] == 12 and s["first_try_rate"] == 1.0
    assert 0 < s["mean_grounding"] <= 1 and "mean grounding" in to_markdown(s)


def test_cli_end_to_end(settings, monkeypatch, capsys):
    from qpg import cli
    from qpg.embeddings import HashingEmbedder
    from qpg.service import Service

    monkeypatch.setattr(cli, "Service", lambda: Service(settings, embedder=HashingEmbedder(), llm=FakeLLM()))
    assert cli.main(["ingest", "--name", "Networks", "--syllabus", str(EXAMPLES / "sample_syllabus.txt"),
                     "--textbook", str(EXAMPLES / "sample_notes.txt")]) == 0
    out = capsys.readouterr().out
    assert "12 of 12 topics covered" in out
    subject = out.split("Ready: ")[1].split()[0]
    assert cli.main(["ask", subject, "--unit", "1", "--marks", "2", "--count", "2"]) == 0
    assert capsys.readouterr().out.count("[2M,") == 2
    assert cli.main(["ask", "does-not-exist", "--marks", "2"]) == 1


def test_paper_file_survives_concurrent_reads_and_writes(service, subject_id):
    import threading

    paper = service.create_paper(subject_id)
    errors, stop = [], threading.Event()

    def reader():
        while not stop.is_set():
            try:
                service.store.get_paper(paper.id)
            except Exception as exc:  # pragma: no cover - only on failure
                errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    try:
        for i in range(200):
            paper.progress = i
            service.store.save_paper(paper)
    finally:
        stop.set()
        for t in threads:
            t.join()
    assert errors == [] and service.store.get_paper(paper.id).progress == 199


def test_weakly_grounded_question_is_retried(service, subject_id):
    import dataclasses

    replies = iter([
        {"question": "Describe how volcanoes erupt and form igneous rock layers."},
        {"question": "Explain how bit stuffing marks where a frame starts and ends."},
    ])
    service.llm = FakeLLM(lambda prompt: next(replies))
    service.settings = dataclasses.replace(service.settings, min_grounding=0.15)
    q = service.generator(subject_id, seed=1).generate("Framing", marks=5, unit=2)
    assert q.attempts == 2 and "bit stuffing" in q.text and q.grounding >= 0.15
