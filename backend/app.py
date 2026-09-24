"""FastAPI backend for the DengueCare RAG Assistant (Project 1).

Supports three modes (set RAG_MODE in .env):
  - kb    : REAL AWS path -> Amazon Bedrock Knowledge Base (retrieve_and_generate)
  - local : Bedrock embeddings + in-code vector retrieval + Bedrock generation
  - mock  : No AWS, deterministic answers (guaranteed working demo)

Endpoints:
  GET  /            -> serves the web UI
  GET  /health      -> health + current mode
  GET  /reports     -> list available sample reports (from sample_reports/)
  POST /upload      -> upload a report (PDF or TXT); indexes it (local/mock)
  POST /ask         -> ask a question; returns grounded answer + evidence
"""
import logging
import os
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config, pii, rag

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app")

app = FastAPI(title="DengueCare RAG Assistant")

# CORS so a separately hosted UI (e.g. calling API Gateway) can reach the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(config.UPLOAD_DIR, exist_ok=True)

_ROOT = os.path.dirname(os.path.dirname(__file__))
_FRONTEND_DIR = os.path.join(_ROOT, "frontend")
_SAMPLES_DIR = os.path.join(_ROOT, "sample_reports")


def _extract_text(path: str, filename: str) -> str:
    """Extract text from a TXT or PDF file."""
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(path)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": config.RAG_MODE,
        "region": config.AWS_REGION,
        "kb_id": config.KB_ID or None,
        "guardrail": bool(config.GUARDRAIL_ID),
    }


@app.get("/reports")
def list_reports():
    """List bundled synthetic sample reports the user can pick from."""
    if not os.path.isdir(_SAMPLES_DIR):
        return {"reports": []}
    names = sorted(
        f for f in os.listdir(_SAMPLES_DIR) if f.lower().endswith((".txt", ".pdf"))
    )
    return {"reports": names}


@app.post("/select-sample")
async def select_sample(name: str = Form(...)):
    """Index a bundled sample report (for local/mock modes)."""
    path = os.path.join(_SAMPLES_DIR, os.path.basename(name))
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Sample report not found.")
    text = _extract_text(path, name)
    doc_id = uuid.uuid4().hex[:12]
    n_chunks = rag.store.index_document(doc_id, text)
    logger.info("Indexed sample %s as doc %s (%d chunks)", name, doc_id, n_chunks)
    return {"doc_id": doc_id, "filename": name, "chunks": n_chunks}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    name = file.filename or "report.txt"
    if not name.lower().endswith((".txt", ".pdf")):
        raise HTTPException(status_code=400, detail="Only .txt and .pdf are supported.")

    doc_id = uuid.uuid4().hex[:12]
    stored_path = os.path.join(config.UPLOAD_DIR, f"{doc_id}_{name}")
    content = await file.read()
    with open(stored_path, "wb") as fh:
        fh.write(content)

    text = _extract_text(stored_path, name)
    if not text.strip():
        raise HTTPException(status_code=422, detail="No readable text found in file.")

    n_chunks = rag.store.index_document(doc_id, text)
    logger.info("Indexed upload %s as doc %s (%d chunks)", name, doc_id, n_chunks)
    return {"doc_id": doc_id, "filename": name, "chunks": n_chunks}


@app.post("/ask")
async def ask(question: str = Form(...), doc_id: str = Form(default="")):
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")

    # ---- KB MODE: managed AWS RAG (no local doc_id needed) ----------------
    if config.USE_KB:
        if not config.KB_ID:
            raise HTTPException(status_code=500, detail="KB_ID not configured.")
        from . import kb_client

        result = kb_client.retrieve_and_generate(question)
        return {
            "answer": result["answer"],
            "evidence": result["evidence"],
            "pii_redacted": [],  # Guardrails handle safety in KB mode
            "mode": config.RAG_MODE,
        }

    # ---- LOCAL / MOCK MODE: in-code retrieval -----------------------------
    if not doc_id or not rag.store.has(doc_id):
        raise HTTPException(
            status_code=404, detail="Select or upload a report first (missing doc_id)."
        )

    hits = rag.store.retrieve(doc_id, question)
    context = "\n\n".join(h["text"] for h in hits)

    # Responsible AI: redact PII before the model call.
    safe_context, pii_found = pii.redact(context)
    if pii_found:
        logger.info("Redacted PII types before model call: %s", pii_found)

    answer = rag.bedrock_client.generate_answer(question, safe_context)
    return {
        "answer": answer,
        "evidence": [
            {"score": round(h["score"], 4), "text": h["text"], "source": "local"}
            for h in hits
        ],
        "pii_redacted": pii_found,
        "mode": config.RAG_MODE,
    }


@app.get("/")
def index():
    return FileResponse(os.path.join(_FRONTEND_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")
