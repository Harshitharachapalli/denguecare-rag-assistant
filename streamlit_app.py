"""Streamlit interface for the Dengue Report Assistant.

Lets a user pick or upload a dengue test report and ask questions about it.
Answers are grounded in the report content using retrieval plus a language model.

Run:
    streamlit run streamlit_app.py
"""
import os
import uuid

import streamlit as st

# Allow the same configuration to work locally (.env) and on Streamlit Cloud
# (Secrets) by copying any string secrets into the environment first.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:  # noqa: BLE001
    pass

from backend import bedrock_client, config, kb_client, pii, rag  # noqa: E402

st.set_page_config(page_title="Dengue Report Assistant", layout="wide")

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


def summarize_report(text: str) -> dict:
    """Pull a few key fields from a report for a quick summary panel.

    Returns a dict of label -> value (only fields that were found).
    """
    fields = {
        "Report ID": r"Report ID\s*:\s*(.+)",
        "Collection date": r"Collection Date\s*:\s*(.+)",
        "NS1 antigen": r"NS1 Antigen\s*:\s*(\w+)",
        "IgM antibody": r"IgM Antibody\s*:\s*(\w+)",
        "IgG antibody": r"IgG Antibody\s*:\s*(\w+)",
        "Platelet count": r"Platelet Count\s*:\s*([\d,]+ /uL[^\n(]*)",
        "WBC count": r"Total WBC Count\s*:\s*([\d,]+ /uL[^\n(]*)",
    }
    import re

    found = {}
    for label, pat in fields.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            found[label] = m.group(1).strip()
    return found


# Session state
for key, default in {
    "doc_id": None,
    "doc_name": None,
    "doc_text": "",
    "history": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# Minimal, restrained styling.
st.markdown(
    """
    <style>
      .block-container { max-width: 1080px; padding-top: 2rem; }
      .answer {
        background: #f7f9fc;
        border: 1px solid #e4e8ef;
        border-radius: 6px;
        padding: 14px 16px;
        line-height: 1.55;
        white-space: pre-wrap;
      }
      .evi {
        border-left: 3px solid #d0d7e2;
        padding: 6px 12px;
        margin-bottom: 8px;
        color: #46505f;
        font-size: .88rem;
        white-space: pre-wrap;
      }
      .muted { color: #6b7280; font-size: .9rem; }
      #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Sidebar: choose a report.
with st.sidebar:
    st.markdown("### Report")

    if not config.USE_KB:
        samples = list_samples()
        picked = st.selectbox("Select a sample report", ["Select..."] + samples)
        if st.button("Load report", use_container_width=True):
            if picked and picked != "Select...":
                text = read_sample(picked)
                did = uuid.uuid4().hex[:12]
                rag.store.index_document(did, text)
                st.session_state.doc_id = did
                st.session_state.doc_name = picked
                st.session_state.doc_text = text
                st.session_state.history = []
            else:
                st.warning("Please select a report.")

        st.markdown("Or upload your own")
        up = st.file_uploader("Upload a report", type=["txt", "pdf"],
                              label_visibility="collapsed")
        if up is not None and st.button("Load uploaded file", use_container_width=True):
            text = extract_uploaded(up)
            if text.strip():
                did = uuid.uuid4().hex[:12]
                rag.store.index_document(did, text)
                st.session_state.doc_id = did
                st.session_state.doc_name = up.name
                st.session_state.doc_text = text
                st.session_state.history = []
            else:
                st.error("Could not read any text from that file.")
    else:
        st.markdown(
            '<span class="muted">Reports are indexed in the knowledge base.'
            " Ask a question directly.</span>",
            unsafe_allow_html=True,
        )

    if st.session_state.doc_name:
        st.markdown("---")
        st.markdown(f"**Loaded:** {st.session_state.doc_name}")

    st.markdown("---")
    st.markdown('<span class="muted">Synthetic sample data. Not medical advice.</span>',
                unsafe_allow_html=True)


# Main area.
st.title("Dengue Report Assistant")
st.markdown(
    '<p class="muted">Ask questions about a dengue test report and get answers '
    "based on the report's contents.</p>",
    unsafe_allow_html=True,
)

col_main, col_report = st.columns([3, 2], gap="large")

with col_report:
    st.markdown("#### Report")
    if st.session_state.doc_name:
        summary = summarize_report(st.session_state.doc_text)
        if summary:
            st.markdown("**Summary**")
            top = st.columns(3)
            for i, key in enumerate(["NS1 antigen", "IgM antibody", "IgG antibody"]):
                if key in summary:
                    top[i].metric(key, summary[key])
            bottom = st.columns(2)
            for i, key in enumerate(["Platelet count", "WBC count"]):
                if key in summary:
                    bottom[i].metric(key, summary[key])
            meta = [f"{k}: {summary[k]}" for k in ("Report ID", "Collection date")
                    if k in summary]
            if meta:
                st.markdown(
                    f'<span class="muted">{" &nbsp;|&nbsp; ".join(meta)}</span>',
                    unsafe_allow_html=True,
                )
            st.markdown("**Full report**")
        st.text_area(
            "Report",
            st.session_state.doc_text,
            height=300,
            label_visibility="collapsed",
        )
    elif config.USE_KB:
        st.markdown('<span class="muted">Stored in the knowledge base.</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="muted">No report loaded yet. Choose one on the left.</span>',
                    unsafe_allow_html=True)

with col_main:
    st.markdown("#### Question")

    sample_questions = [
        "What is the platelet count?",
        "Is the NS1 antigen positive?",
        "Is there an active dengue infection?",
        "What is the total WBC count?",
        "What does the interpretation say?",
    ]
    chosen = st.selectbox("Sample questions", ["Type your own..."] + sample_questions)
    default_q = "" if chosen == "Type your own..." else chosen

    question = st.text_input(
        "Question",
        value=default_q,
        placeholder="For example: What is the platelet count?",
        label_visibility="collapsed",
    )
    ask = st.button("Ask", type="primary")

    ready = config.USE_KB or st.session_state.doc_id is not None

    if ask:
        if not question.strip():
            st.warning("Please enter a question.")
        elif not ready:
            st.warning("Load a report first.")
        else:
            with st.spinner("Looking through the report..."):
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
                        {"score": round(h["score"], 4), "text": h["text"]}
                        for h in hits
                    ]
            st.session_state.history.insert(0, (question, answer, evidence, pii_found))

    for q, a, evi, pii_found in st.session_state.history:
        st.markdown(f"**{q}**")
        st.markdown(f'<div class="answer">{a}</div>', unsafe_allow_html=True)
        if pii_found:
            st.markdown(
                f'<span class="muted">Sensitive fields were masked before '
                f'processing: {", ".join(pii_found)}.</span>',
                unsafe_allow_html=True,
            )
        with st.expander("Show the report sections used"):
            for c in evi:
                safe = c["text"].replace("<", "&lt;")
                st.markdown(f'<div class="evi">{safe}</div>', unsafe_allow_html=True)
        st.markdown("")
