"""Central configuration loaded from environment variables (.env).

RAG_MODE selects how answers are produced:
  - "kb"    : REAL AWS path. Uses Amazon Bedrock Knowledge Base
              (bedrock-agent-runtime.retrieve_and_generate). Needs KB_ID.
  - "local" : Runs embeddings + retrieval in-code, calls Bedrock for the
              final answer. Good when you have Bedrock model access but no KB yet.
  - "mock"  : No AWS at all. Deterministic demo answers. Guarantees a running demo.
"""
import os
from dotenv import load_dotenv

load_dotenv()

RAG_MODE = os.getenv("RAG_MODE", "mock").strip().lower()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Bedrock model IDs
BEDROCK_TEXT_MODEL_ID = os.getenv(
    "BEDROCK_TEXT_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"
)
BEDROCK_EMBED_MODEL_ID = os.getenv(
    "BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0"
)

# Bedrock Knowledge Base (used only when RAG_MODE=kb)
KB_ID = os.getenv("KB_ID", "").strip()

# Optional Bedrock Guardrail
GUARDRAIL_ID = os.getenv("GUARDRAIL_ID", "").strip()
GUARDRAIL_VERSION = os.getenv("GUARDRAIL_VERSION", "DRAFT").strip()

# RAG settings (used in local mode)
CHUNK_SIZE = 600          # characters per chunk
CHUNK_OVERLAP = 100       # overlap between chunks
TOP_K = 3                 # number of chunks retrieved per question

# Where uploaded reports are stored (local stand-in for Amazon S3)
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")

# Convenience flags
USE_MOCK = RAG_MODE == "mock"
USE_KB = RAG_MODE == "kb"
