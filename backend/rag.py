"""Core RAG pipeline: chunking, in-memory vector store, retrieval.

The in-memory vector store is a local stand-in for Amazon S3 Vectors /
OpenSearch Serverless. It stores (chunk_text, embedding) per document and
retrieves the top-K most similar chunks for a question using cosine similarity.
"""
import math

from . import bedrock_client, config


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping character chunks."""
    text = text.strip()
    if not text:
        return []
    chunks = []
    step = config.CHUNK_SIZE - config.CHUNK_OVERLAP
    for start in range(0, len(text), step):
        chunk = text[start : start + config.CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class VectorStore:
    """Simple per-document in-memory vector store."""

    def __init__(self) -> None:
        # doc_id -> list of {"text": str, "embedding": list[float]}
        self._docs: dict[str, list[dict]] = {}

    def index_document(self, doc_id: str, text: str) -> int:
        chunks = chunk_text(text)
        records = [{"text": c, "embedding": bedrock_client.embed(c)} for c in chunks]
        self._docs[doc_id] = records
        return len(records)

    def has(self, doc_id: str) -> bool:
        return doc_id in self._docs

    def retrieve(self, doc_id: str, question: str, top_k: int = config.TOP_K):
        records = self._docs.get(doc_id, [])
        if not records:
            return []
        q_emb = bedrock_client.embed(question)
        scored = [
            {"text": r["text"], "score": _cosine(q_emb, r["embedding"])}
            for r in records
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


# Single shared store for the app lifetime.
store = VectorStore()
