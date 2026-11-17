import random
from pathlib import Path

import pytest

from qpg.config import Settings
from qpg.embeddings import HashingEmbedder
from qpg.llm import FakeLLM
from qpg.service import Service

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def settings(tmp_path):
    # The hashing embedder scores lower than MiniLM, so the coverage threshold is lower too.
    return Settings(data_dir=tmp_path / "data", embedder="hashing", min_similarity=0.08, chunk_chars=700,
                    min_grounding=0.02)


@pytest.fixture
def service(settings):
    return Service(settings, embedder=HashingEmbedder(), llm=FakeLLM())


@pytest.fixture
def subject_id(service):
    meta = service.store.create_subject("Computer Networks")
    service.ingest(meta.id, [EXAMPLES / "sample_notes.txt"], syllabus_path=EXAMPLES / "sample_syllabus.txt")
    return meta.id


@pytest.fixture
def rng():
    return random.Random(7)
