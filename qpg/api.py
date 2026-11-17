"""HTTP API. Run with:  uvicorn qpg.api:app --port 8000"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from qpg import __version__
from qpg.extract import SUPPORTED
from qpg.generate import NoSourceMaterial
from qpg.llm import LLMError
from qpg.models import Blueprint, Paper, Question, SubjectMeta
from qpg.service import Service


class QuestionRequest(BaseModel):
    marks: int = Field(gt=0, le=100)
    count: int = Field(default=1, gt=0, le=20)
    unit: Optional[int] = None
    topic: Optional[str] = Field(default=None, max_length=200)
    bloom: Optional[List[str]] = None


class PaperRequest(BaseModel):
    blueprint: Optional[Blueprint] = None


def _safe_name(filename: str) -> str:
    name = Path(filename or "file").name  # drop any directory part the client sent
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(". ") or "file"
    if Path(name).suffix.lower() not in SUPPORTED:
        raise HTTPException(400, f"'{name}' is not supported. Upload PDF, DOCX, TXT or MD files.")
    return name


def create_app(service: Optional[Service] = None) -> FastAPI:
    svc = service or Service()
    app = FastAPI(title="AI Question Paper Generator", version=__version__)
    app.add_middleware(
        CORSMiddleware, allow_origins=list(svc.settings.cors_origins), allow_methods=["*"], allow_headers=["*"]
    )

    async def save_upload(upload: UploadFile, folder: Path) -> Path:
        target = folder / _safe_name(upload.filename)
        limit = svc.settings.max_upload_mb * 1024 * 1024
        written = 0
        with open(target, "wb") as handle:
            while True:
                block = await upload.read(1024 * 1024)
                if not block:
                    break
                written += len(block)
                if written > limit:
                    handle.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(413, f"'{target.name}' is larger than {svc.settings.max_upload_mb} MB.")
                handle.write(block)
        return target

    def guard(fn, *args, **kwargs):
        """Translate domain errors into HTTP status codes."""
        try:
            return fn(*args, **kwargs)
        except KeyError as exc:
            raise HTTPException(404, str(exc).strip("'\""))
        except NoSourceMaterial as exc:
            raise HTTPException(422, str(exc))
        except LLMError as exc:
            raise HTTPException(503, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "version": __version__,
            "llm_model": svc.settings.llm_model,
            "llm_available": svc.llm.available(),
            "embedder": svc.settings.embedder,
        }

    @app.get("/api/subjects", response_model=List[SubjectMeta])
    def list_subjects():
        return svc.store.list_subjects()

    @app.post("/api/subjects", response_model=SubjectMeta, status_code=202)
    async def create_subject(
        name: str = Form(..., min_length=1, max_length=120),
        syllabus_text: str = Form(""),
        syllabus: Optional[UploadFile] = File(None),
        textbooks: List[UploadFile] = File(...),
    ):
        if not syllabus_text.strip() and syllabus is None:
            raise HTTPException(400, "Upload a syllabus file or paste the syllabus text.")
        meta = svc.store.create_subject(name)
        uploads = svc.store.subject_dir(meta.id) / "uploads"
        try:
            syllabus_path = await save_upload(syllabus, uploads) if syllabus is not None else None
            book_paths = [await save_upload(book, uploads) for book in textbooks]
        except HTTPException:
            svc.store.delete_subject(meta.id)
            raise
        svc.ingest_in_background(
            meta.id, textbooks=book_paths, syllabus_path=syllabus_path, syllabus_text=syllabus_text
        )
        return meta

    @app.get("/api/subjects/{subject_id}", response_model=SubjectMeta)
    def get_subject(subject_id: str):
        return guard(svc.store.get_meta, subject_id)

    @app.delete("/api/subjects/{subject_id}", status_code=204)
    def delete_subject(subject_id: str):
        guard(svc.store.delete_subject, subject_id)

    @app.get("/api/subjects/{subject_id}/search")
    def search(subject_id: str, q: str, k: int = 5):
        """Show which textbook passages a topic retrieves. Useful for checking coverage."""
        subject = guard(svc.store.load_subject, subject_id, svc.embedder)
        hits = subject.retriever.search(q, k=max(1, min(k, 20)))
        return [
            {"rank": h.rank, "similarity": round(h.similarity, 4), "source": h.chunk.source,
             "page_start": h.chunk.page_start, "page_end": h.chunk.page_end, "text": h.chunk.text}
            for h in hits
        ]

    @app.post("/api/subjects/{subject_id}/questions", response_model=List[Question])
    def generate_questions(subject_id: str, req: QuestionRequest):
        return guard(
            svc.generate_questions, subject_id, marks=req.marks, count=req.count,
            unit=req.unit, topic=(req.topic or "").strip() or None, bloom=req.bloom,
        )

    @app.get("/api/subjects/{subject_id}/blueprint", response_model=Blueprint)
    def get_blueprint(subject_id: str):
        return guard(svc.default_blueprint, subject_id)

    @app.post("/api/subjects/{subject_id}/papers", response_model=Paper, status_code=202)
    def create_paper(subject_id: str, req: PaperRequest):
        paper = guard(svc.create_paper, subject_id, req.blueprint)
        svc.run_paper_in_background(paper)
        return paper

    @app.get("/api/papers/{paper_id}", response_model=Paper)
    def get_paper(paper_id: str):
        return guard(svc.store.get_paper, paper_id)

    @app.get("/api/papers/{paper_id}/download")
    def download_paper(paper_id: str):
        # The file is looked up by id inside the papers folder. The client never
        # supplies a path, so it cannot read arbitrary files from the server.
        paper = guard(svc.store.get_paper, paper_id)
        path = svc.store.paper_docx(paper_id)
        if paper.status != "ready" or not path.exists():
            raise HTTPException(409, "This paper is not ready to download yet.")
        return FileResponse(
            path,
            filename=f"{re.sub(r'[^A-Za-z0-9]+', '-', paper.subject_name).strip('-') or 'paper'}-{paper.id}.docx",
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    # In the Docker image the built React app is served by the same server.
    static_dir = Path(os.environ.get("QPG_STATIC_DIR", "frontend/dist"))
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
    return app


app = create_app()
