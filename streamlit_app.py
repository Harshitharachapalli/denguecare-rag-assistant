"""Streamlit UI for the DengueCare RAG Assistant (Project 1).

Reuses the same backend logic as the FastAPI app:
  - backend.rag        : chunking + in-memory vector store + retrieval
  - backend.bedrock_client : embeddings + generation (Bedrock / mock)
  - backend.kb_client  : Amazon Bedrock Knowledge Base (real managed RAG)
  - backend.pii        : PII redaction (Responsible AI)

Run locally:
    streamlit run streamlit_app.py
"""
import os
import uuid

import streamlit as st

# Bridge Streamlit Cloud secrets -> environment BEFORE importing backend.config,
# so the same config code works locally (.env) and on Streamlit Cloud (Secrets).
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:  # noqa: BLE001
    pass

from backend import bedrock_client, config, kb_client, pii, rag  # noqa: E402

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="DengueCare RAG Assistant",
    page_icon="🩺",
    layout="wide",
)

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(ROOT, "sample_reports")


def list_samples() -> list[str]:
    if not os.path.isdir(SAMPLES_DIR):
        return []
    return sorted(
        f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".txt", ".pdf"))
    )


def read_sample(name: str) -> str:
    path = os.path.join(SAMPLES_DIR, os.path.basename(name))
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def extract_uploaded(file) -> str:
    if file.name.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(file)
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    return file.read().decode("utf-8", errors="ignore")


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""
if "history" not in st.session_state:
    st.session_state.history = []  # list of (question, answer, evidence, pii)


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
      .app-title { font-size: 2rem; font-weight: 700; margin-bottom: .1rem; }
      .app-sub { color: #64748b; margin-bottom: 1rem; }
      .answer-box {
        background: #eff6ff; border-left: 5px solid #2563eb;
        padding: 16px 18px; border-radius: 10px; font-size: 1.02rem;
        line-height: 1.55; white-space: pre-wrap;
      }
      .evi {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
        padding: 10px 12px; margin-bottom: 8px; font-size: .85rem;
        white-space: pre-wrap;
      }
      .evi .score { color: #059669; font-weight: 600; }
      .pill {
        display:inline-block; background:#dbeafe; color:#1e3a8a;
        padding:3px 12px; border-radius:999px; font-size:.8rem; font-weight:600;
      }
      .warnpill {
        display:inline-block; background:#fef3c7; color:#b45309;
        padding:3px 12px; border-radius:999px; font-size:.8rem; font-weight:600;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Sidebar: config + report selection
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Setup")
    mode_label = {
        "mock": "Mock (no AWS)",
        "local": "Local (Bedrock model)",
        "kb": "Knowledge Base (full AWS)",
    }.get(config.RAG_MODE, config.RAG_MODE)
    st.markdown(f"**Mode:** `{config.RAG_MODE}` — {mode_label}")
    st.markdown(f"**Region:** `{config.AWS_REGION}`")
    if config.USE_KB and config.KB_ID:
        st.markdown(f"**KB ID:** `{config.KB_ID}`")
    st.caption("Change RAG_MODE in your .env file to switch between demo and real AWS.")

    st.divider()
    st.subheader("📄 Choose a report")

    if not config.USE_KB:
        samples = list_samples()
        picked = st.selectbox("Bundled synthetic reports", ["— select —"] + samples)
        if st.button("Use this report", use_container_width=True):
            if picked and picked != "— select —":
                text = read_sample(picked)
                did = uuid.uuid4().hex[:12]
                n = rag.store.index_document(did, text)
                st.session_state.doc_id = did
                st.session_state.doc_name = picked
                st.session_state.doc_text = text
                st.session_state.history = []
                st.success(f"Indexed '{picked}' into {n} chunks.")
            else:
                st.warning("Pick a report first.")

        st.markdown("**or upload your own**")
        up = st.file_uploader("Upload .txt or .pdf", type=["txt", "pdf"])
        if up is not None and st.button("Upload & index", use_container_width=True):
            text = extract_uploaded(up)
            if text.strip():
                did = uuid.uuid4().hex[:12]
                n = rag.store.index_document(did, text)
                st.session_state.doc_id = did
                st.session_state.doc_name = up.name
                st.session_state.doc_text = text
                st.session_state.history = []
                st.success(f"Indexed '{up.name}' into {n} chunks.")
            else:
                st.error("No readable text found in the file.")
    else:
        st.info(
            "Knowledge Base mode: reports are already indexed from S3. "
            "Ask a question directly."
        )

    st.divider()
    st.caption("Synthetic data only. Not medical advice.")


# --------------------------------------------------------------------------- #
# Main area
# --------------------------------------------------------------------------- #
st.markdown('<div class="app-title">🩺 DengueCare RAG Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-sub">Ask questions about a dengue test report. Answers are '
    "grounded in the report using Retrieval-Augmented Generation + Amazon Bedrock."
    "</div>",
    unsafe_allow_html=True,
)

col_main, col_report = st.columns([3, 2], gap="large")

with col_report:
    st.subheader("Selected report")
    if st.session_state.doc_name:
        st.markdown(f"**{st.session_state.doc_name}**")
        st.text_area(
            "Report content",
            st.session_state.doc_text,
            height=360,
            label_visibility="collapsed",
        )
    elif config.USE_KB:
        st.caption("Reports live in the Knowledge Base (from S3).")
    else:
        st.caption("No report selected yet. Pick or upload one in the sidebar.")

with col_main:
    st.subheader("Ask a question")

    example_qs = [
        "What is the platelet count?",
        "Is the NS1 antigen positive?",
        "Is there an active dengue infection?",
        "What does the interpretation say?",
    ]
    ex = st.selectbox("Example questions (optional)", ["— type your own —"] + example_qs)
    default_q = "" if ex == "— type your own —" else ex
    question = st.text_input("Your question", value=default_q, placeholder="e.g. What is the platelet count?")

    ask = st.button("🔍 Ask", type="primary")

    ready = config.USE_KB or st.session_state.doc_id is not None

    if ask:
        if not question.strip():
            st.warning("Type a question first.")
        elif not ready:
            st.warning("Select or upload a report in the sidebar first.")
        else:
            with st.spinner("Retrieving and generating a grounded answer..."):
                if config.USE_KB:
                    result = kb_client.retrieve_and_generate(question)
                    answer = result["answer"]
                    evidence = result["evidence"]
                    pii_found = []
                else:
                    hits = rag.store.retrieve(st.session_state.doc_id, question)
                    context = "\n\n".join(h["text"] for h in hits)
                    safe_context, pii_found = pii.redact(context)
                    answer = bedrock_client.generate_answer(question, safe_context)
                    evidence = [
                        {"score": round(h["score"], 4), "text": h["text"], "source": "local"}
                        for h in hits
                    ]
            st.session_state.history.insert(0, (question, answer, evidence, pii_found))

    # Render conversation history (most recent first)
    for q, a, evi, pii_found in st.session_state.history:
        st.markdown(f"**Q: {q}**")
        st.markdown(f'<div class="answer-box">{a}</div>', unsafe_allow_html=True)
        if pii_found:
            st.markdown(
                f'<span class="warnpill">⚠ PII redacted before model call: '
                f'{", ".join(pii_found)}</span>',
                unsafe_allow_html=True,
            )
        with st.expander(f"Evidence — {len(evi)} retrieved chunk(s)"):
            for i, c in enumerate(evi, 1):
                score = f" · score {c['score']}" if c.get("score") is not None else ""
                src = f"<br><small>source: {c.get('source','')}</small>" if c.get("source") else ""
                safe = c["text"].replace("<", "&lt;")
                st.markdown(
                    f'<div class="evi"><span class="score">#{i}{score}</span>{src}'
                    f"<div>{safe}</div></div>",
                    unsafe_allow_html=True,
                )
        st.divider()
