"""Amazon Bedrock client wrapper with a mock fallback.

If USE_MOCK is True (or AWS is unreachable), the app still works using
deterministic mock embeddings and a rule-based answer. This guarantees a
working demo even without Bedrock model access.
"""
import hashlib
import json
import logging
import math

from . import config

logger = logging.getLogger("bedrock")

_bedrock = None


def _client():
    """Lazily create the boto3 Bedrock runtime client."""
    global _bedrock
    if _bedrock is None:
        import boto3

        _bedrock = boto3.client("bedrock-runtime", region_name=config.AWS_REGION)
    return _bedrock


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def _mock_embedding(text: str, dim: int = 256) -> list[float]:
    """Deterministic pseudo-embedding from a hash. Good enough for demo retrieval."""
    vec = [0.0] * dim
    tokens = text.lower().split()
    for tok in tokens:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def embed(text: str) -> list[float]:
    """Return an embedding vector for the given text."""
    if config.USE_MOCK:
        return _mock_embedding(text)
    try:
        body = json.dumps({"inputText": text})
        resp = _client().invoke_model(
            modelId=config.BEDROCK_EMBED_MODEL_ID,
            body=body,
            accept="application/json",
            contentType="application/json",
        )
        payload = json.loads(resp["body"].read())
        return payload["embedding"]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Embedding via Bedrock failed (%s); using mock.", exc)
        return _mock_embedding(text)


# --------------------------------------------------------------------------- #
# Text generation
# --------------------------------------------------------------------------- #
def _mock_answer(question: str, context: str) -> str:
    """Rule-based grounded answer used when Bedrock is unavailable."""
    snippet = context.strip().replace("\n", " ")
    if len(snippet) > 500:
        snippet = snippet[:500] + "..."
    return (
        "[MOCK MODE - no AWS call was made]\n\n"
        f"Based on the retrieved report content, here is what is relevant to "
        f'your question "{question}":\n\n{snippet}\n\n'
        "Enable real Bedrock by setting RAG_MODE=local (or kb) in .env once "
        "model access is approved."
    )


def generate_answer(question: str, context: str) -> str:
    """Generate a grounded answer from the retrieved context."""
    prompt = (
        "You are a careful healthcare-information assistant. Answer the user's "
        "question USING ONLY the context from the dengue report below. If the "
        "answer is not in the context, say you cannot find it in the report. "
        "Do not give medical advice; only report what the document states.\n\n"
        f"=== REPORT CONTEXT ===\n{context}\n\n"
        f"=== QUESTION ===\n{question}\n\n=== ANSWER ==="
    )

    if config.USE_MOCK:
        return _mock_answer(question, context)

    try:
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 600,
                "temperature": 0.2,
                "messages": [{"role": "user", "content": prompt}],
            }
        )
        resp = _client().invoke_model(
            modelId=config.BEDROCK_TEXT_MODEL_ID,
            body=body,
            accept="application/json",
            contentType="application/json",
        )
        payload = json.loads(resp["body"].read())
        return payload["content"][0]["text"]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Generation via Bedrock failed (%s); using mock.", exc)
        return _mock_answer(question, context)
