import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from qpg.api import create_app  # noqa: E402
from tests.conftest import EXAMPLES  # noqa: E402


@pytest.fixture
def client(service):
    return TestClient(create_app(service))


def wait_for(client, url, done=("ready", "failed"), timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(url).json()
        if body["status"] in done:
            return body
        time.sleep(0.05)
    raise AssertionError(f"{url} did not finish")


def upload(client):
    files = [
        ("syllabus", ("syllabus.txt", (EXAMPLES / "sample_syllabus.txt").read_bytes(), "text/plain")),
        ("textbooks", ("notes.txt", (EXAMPLES / "sample_notes.txt").read_bytes(), "text/plain")),
    ]
    response = client.post("/api/subjects", data={"name": "Computer Networks"}, files=files)
    assert response.status_code == 202
    return wait_for(client, f"/api/subjects/{response.json()['id']}")


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["llm_available"] is True


def test_full_flow(client):
    subject = upload(client)
    assert subject["status"] == "ready" and len(subject["syllabus"]["units"]) == 3
    sid = subject["id"]
    assert [s["id"] for s in client.get("/api/subjects").json()] == [sid]

    hits = client.get(f"/api/subjects/{sid}/search", params={"q": "congestion control"}).json()
    assert "congestion" in hits[0]["text"].lower()

    r = client.post(f"/api/subjects/{sid}/questions", json={"marks": 2, "count": 2, "unit": 2})
    assert r.status_code == 200 and len(r.json()) == 2 and r.json()[0]["unit"] == 2

    r = client.post(f"/api/subjects/{sid}/questions", json={"marks": 2, "topic": "zebrafish xylophone"})
    assert r.status_code == 422

    paper = client.post(f"/api/subjects/{sid}/papers", json={}).json()
    paper = wait_for(client, f"/api/papers/{paper['id']}")
    assert paper["status"] == "ready"
    download = client.get(f"/api/papers/{paper['id']}/download")
    assert download.status_code == 200 and download.content[:2] == b"PK"

    assert client.delete(f"/api/subjects/{sid}").status_code == 204
    assert client.get(f"/api/subjects/{sid}").status_code == 404


def test_validation_and_safety(client):
    assert client.get("/api/subjects/nope").status_code == 404
    assert client.get("/api/papers/..%2F..%2Fetc%2Fpasswd/download").status_code == 404
    assert client.post("/api/subjects/nope/questions", json={"marks": 0}).status_code == 422
    bad = client.post("/api/subjects", data={"name": "X", "syllabus_text": "Unit 1 A\nB, C, D"},
                      files=[("textbooks", ("run.exe", b"MZ", "application/octet-stream"))])
    assert bad.status_code == 400
    assert client.get("/api/subjects").json() == []  # the half-created subject was cleaned up
