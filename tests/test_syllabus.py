from qpg.llm import FakeLLM
from qpg.syllabus import parse_heuristic, parse_syllabus, parse_with_llm, roman_to_int, split_topics


def names(unit):
    return [t.name for t in unit.topics]


def test_roman_numerals():
    assert [roman_to_int(x) for x in ["I", "iv", "IX", "XII"]] == [1, 4, 9, 12]


def test_roman_units_with_dash_separated_topics():
    text = """UNIT I - FOUNDATIONS (9 Hours)
Layered architecture - OSI reference model - TCP/IP model

UNIT II - DATA LINK LAYER 9 Hrs
Framing - Error detection with CRC
"""
    syllabus = parse_heuristic(text)
    assert [u.number for u in syllabus.units] == [1, 2]
    assert syllabus.units[0].title == "FOUNDATIONS"
    assert names(syllabus.units[0]) == ["Layered architecture", "OSI reference model", "TCP/IP model"]
    assert names(syllabus.units[1]) == ["Framing", "Error detection with CRC"]


def test_module_headers_with_inline_comma_topics():
    text = """Module 1: Process Management: Processes, Threads, CPU scheduling; Deadlocks.
Module 2: Memory: Paging, Segmentation, Virtual memory."""
    syllabus = parse_heuristic(text)
    assert syllabus.units[0].title == "Process Management"
    assert names(syllabus.units[0]) == ["Processes", "Threads", "CPU scheduling", "Deadlocks"]
    assert names(syllabus.units[1]) == ["Paging", "Segmentation", "Virtual memory"]


def test_references_and_outcomes_are_not_topics():
    text = """Unit 1 Basics
Stacks, Queues, Linked lists
Text Books:
1. Some Author, Some Book, Some Publisher
Unit 2 Trees
Binary trees, Heaps
Course Outcomes
Students will be able to do things
"""
    syllabus = parse_heuristic(text)
    topics = [t for u in syllabus.units for t in names(u)]
    assert topics == ["Stacks", "Queues", "Linked lists", "Binary trees", "Heaps"]


def test_bullets_and_numbering_are_stripped():
    assert split_topics("1.1 Arrays\n• Hashing\n2) Graphs (6 hours)\nand Sorting\nIntroduction to Trees\nOverview") == [
        "Arrays", "Hashing", "Graphs", "Sorting", "Introduction to Trees"]


def test_no_unit_headers_falls_back_to_one_unit():
    syllabus = parse_heuristic("Supply and demand, Elasticity, Market structures, Game theory")
    assert len(syllabus.units) == 1
    assert len(syllabus.units[0].topics) == 4


def test_llm_topics_not_in_the_text_are_dropped():
    text = "Photosynthesis and respiration are covered, followed by cell division."
    llm = FakeLLM(lambda prompt: {"units": [
        {"title": "Cells", "topics": ["Photosynthesis", "Cell division", "Quantum computing"]}]})
    syllabus = parse_with_llm(text, llm)
    assert names(syllabus.units[0]) == ["Photosynthesis", "Cell division"]


def test_structured_syllabus_never_calls_the_llm():
    llm = FakeLLM()
    text = "Unit 1 A\nAlpha, Beta, Gamma\nUnit 2 B\nDelta, Epsilon, Zeta"
    assert parse_syllabus(text, llm=llm).parser == "heuristic"
    assert llm.calls == 0


def test_unstructured_syllabus_uses_the_llm_when_it_finds_more_structure():
    text = "This course first studies supply and demand. Later weeks cover elasticity and game theory."
    llm = FakeLLM(lambda prompt: {"units": [
        {"title": "Markets", "topics": ["Supply and demand", "Elasticity"]},
        {"title": "Strategy", "topics": ["Game theory"]}]})
    syllabus = parse_syllabus(text, llm=llm)
    assert syllabus.parser == "llm"
    assert len(syllabus.units) == 2


WRAPPED = """Course Outcome
CO1 Explain how a network is built
UNIT-I Physical Layer Periods: 9
Introduction – Uses – Reference Models – Theoretical Basis For
Communication – Radio
Transmission – Baseband Transmission.

CO1

Unit-II Data Link Layer Periods: 9
Framing - Error Detection
and Correction Codes – Collision-Free Protocols - 802.11 MAC Sublayer Protocol - Flow Control and Buffering–UDP
- Repeaters, Hubs, and
Gateways.

CO2

Lecture Periods: 45 Tutorial Periods: - Total Periods: 45
"""


def test_word_wrapped_dash_separated_syllabus():
    syllabus = parse_heuristic(WRAPPED)
    assert [u.title for u in syllabus.units] == ["Physical Layer", "Data Link Layer"]
    # "Introduction" and "Uses" are filler, not subject matter, and are dropped.
    assert names(syllabus.units[0]) == [
        "Reference Models", "Theoretical Basis For Communication", "Radio Transmission", "Baseband Transmission"]
    assert names(syllabus.units[1]) == [
        "Framing", "Error Detection and Correction Codes", "Collision-Free Protocols",
        "802.11 MAC Sublayer Protocol", "Flow Control and Buffering", "UDP", "Repeaters", "Hubs", "Gateways"]


def test_topic_query_adds_unit_context():
    from qpg.syllabus import topic_query

    assert topic_query("Design Issues", "Data Link Layer") == "Design Issues (Data Link Layer)"
    assert topic_query("Data Link Layer Protocols", "Data Link Layer") == "Data Link Layer Protocols"
    assert topic_query("Paging", "Unit 2") == "Paging"
