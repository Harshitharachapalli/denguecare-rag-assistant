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


import re


def parse_report(text: str) -> dict:
    """Parse a dengue report into structured fields, ranges, and status.

    Returns:
      {
        "meta": {"Report ID":..., "Collection date":..., "Age / Sex":...},
        "rows": [ {test, value, range, status, verdict}, ... ],
        "assessment": str,   # overall plain-language conclusion
      }
    """
    def find(pattern):
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1).strip() if m else None

    def num(s):
        try:
            return float(s.replace(",", "")) if s else None
        except ValueError:
            return None

    meta = {
        "Report ID": find(r"Report ID\s*:\s*(.+)"),
        "Age / Sex": find(r"Age / Sex\s*:\s*(.+)"),
        "Collection date": find(r"Collection Date\s*:\s*(.+)"),
        "Lab": find(r"Reporting Lab\s*:\s*(.+)"),
    }
    meta = {k: v for k, v in meta.items() if v}

    rows = []

    # --- Serology / antigen: qualitative POSITIVE/NEGATIVE ---
    serology = {
        "NS1 Antigen": find(r"NS1 Antigen\s*:\s*(\w+)"),
        "IgM Antibody": find(r"IgM Antibody\s*:\s*(\w+)"),
        "IgG Antibody": find(r"IgG Antibody\s*:\s*(\w+)"),
    }
    for test, val in serology.items():
        if not val:
            continue
        val_up = val.upper()
        if val_up == "POSITIVE":
            # NS1 / IgM positive => active concern (Alert); IgG positive => past exposure (Watch)
            status = "Positive"
            verdict = "Watch" if test == "IgG Antibody" else "Alert"
        else:
            status, verdict = "Negative", "Good"
        rows.append({"test": test, "value": val_up, "range": "Negative", "status": status, "verdict": verdict})

    # --- Numeric CBC values with ranges ---
    plt = num(find(r"Platelet Count\s*:\s*([\d,]+)"))
    if plt is not None:
        if plt < 150000:
            status, verdict = "Low", "Alert"
        elif plt > 410000:
            status, verdict = "High", "Watch"
        else:
            status, verdict = "Normal", "Good"
        rows.append({"test": "Platelet Count", "value": f"{int(plt):,} /uL",
                     "range": "150,000 - 410,000", "status": status, "verdict": verdict})

    wbc = num(find(r"Total WBC Count\s*:\s*([\d,]+)"))
    if wbc is not None:
        if wbc < 4000:
            status, verdict = "Low", "Watch"
        elif wbc > 11000:
            status, verdict = "High", "Watch"
        else:
            status, verdict = "Normal", "Good"
        rows.append({"test": "Total WBC Count", "value": f"{int(wbc):,} /uL",
                     "range": "4,000 - 11,000", "status": status, "verdict": verdict})

    hb = num(find(r"Hemoglobin\s*:\s*([\d.]+)"))
    if hb is not None:
        if hb < 12.0:
            status, verdict = "Low", "Watch"
        elif hb > 16.0:
            status, verdict = "High", "Watch"
        else:
            status, verdict = "Normal", "Good"
        rows.append({"test": "Hemoglobin", "value": f"{hb} g/dL",
                     "range": "12.0 - 16.0", "status": status, "verdict": verdict})

    hct = num(find(r"Hematocrit \(PCV\)\s*:\s*(\d+)"))
    if hct is not None:
        if hct < 36:
            status, verdict = "Low", "Watch"
        elif hct > 50:
            status, verdict = "High", "Watch"
        else:
            status, verdict = "Normal", "Good"
        rows.append({"test": "Hematocrit (PCV)", "value": f"{int(hct)}%",
                     "range": "36 - 50", "status": status, "verdict": verdict})

    # --- Overall assessment from serology + platelets ---
    ns1 = (serology.get("NS1 Antigen") or "").upper()
    igm = (serology.get("IgM Antibody") or "").upper()
    igg = (serology.get("IgG Antibody") or "").upper()
    if ns1 == "POSITIVE" or igm == "POSITIVE":
        assessment = ("Findings are consistent with an ACTIVE dengue infection"
                      + (" with low platelets." if plt and plt < 150000 else "."))
    elif igg == "POSITIVE":
        assessment = "Findings suggest PAST dengue exposure rather than an active infection."
    else:
        assessment = "No serological evidence of dengue in this sample."

    return {"meta": meta, "rows": rows, "assessment": assessment}


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

      table.rpt {
        width: 100%; border-collapse: collapse; font-size: .9rem;
        margin: 6px 0 10px;
      }
      table.rpt th {
        text-align: left; background: #f1f5f9; color: #334155;
        padding: 8px 10px; border-bottom: 2px solid #e2e8f0; font-weight: 600;
      }
      table.rpt td {
        padding: 8px 10px; border-bottom: 1px solid #eef2f7; color: #1f2937;
      }
      .badge {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: .78rem; font-weight: 600;
      }
      .b-good  { background: #dcfce7; color: #15803d; }
      .b-watch { background: #fef9c3; color: #a16207; }
      .b-alert { background: #fee2e2; color: #b91c1c; }
      .assess {
        border-radius: 6px; padding: 10px 14px; font-size: .92rem;
        margin: 8px 0 4px;
      }
      .assess-alert { background: #fef2f2; border: 1px solid #fecaca; color: #991b1b; }
      .assess-watch { background: #fffbeb; border: 1px solid #fde68a; color: #92400e; }
      .assess-good  { background: #f0fdf4; border: 1px solid #bbf7d0; color: #166534; }
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

_BADGE_CLASS = {"Good": "b-good", "Watch": "b-watch", "Alert": "b-alert"}


def recommendations_for(parsed: dict) -> list[str]:
    """Plain-language recommendations derived from the parsed results."""
    recs = []
    by_test = {r["test"]: r for r in parsed["rows"]}

    plt = by_test.get("Platelet Count")
    if plt and plt["status"] == "Low":
        recs.append("Platelets are low. Repeat CBC monitoring is advised.")
    wbc = by_test.get("Total WBC Count")
    if wbc and wbc["status"] == "Low":
        recs.append("WBC count is low, which can occur in dengue. Monitor as advised.")

    if "ACTIVE" in parsed["assessment"]:
        recs.append("Serology indicates active infection. Follow clinical guidance and stay hydrated.")
    elif "PAST" in parsed["assessment"]:
        recs.append("Results point to past exposure, not an active infection.")
    else:
        recs.append("No action indicated from these results in this synthetic scenario.")

    if not any(r["verdict"] != "Good" for r in parsed["rows"]):
        recs.insert(0, "All measured values are within normal ranges.")
    return recs


with col_report:
    st.markdown("#### Report results")
    if st.session_state.doc_name:
        parsed = parse_report(st.session_state.doc_text)

        # Metadata line
        if parsed["meta"]:
            meta_txt = " &nbsp;|&nbsp; ".join(f"{k}: {v}" for k, v in parsed["meta"].items())
            st.markdown(f'<span class="muted">{meta_txt}</span>', unsafe_allow_html=True)

        # Results table
        if parsed["rows"]:
            html = ['<table class="rpt"><tr><th>Test</th><th>Result</th>'
                    '<th>Reference</th><th>Status</th></tr>']
            for r in parsed["rows"]:
                cls = _BADGE_CLASS.get(r["verdict"], "b-watch")
                html.append(
                    f'<tr><td>{r["test"]}</td><td>{r["value"]}</td>'
                    f'<td>{r["range"]}</td>'
                    f'<td><span class="badge {cls}">{r["status"]}</span></td></tr>'
                )
            html.append("</table>")
            st.markdown("".join(html), unsafe_allow_html=True)

        # Overall assessment
        acls = ("assess-alert" if "ACTIVE" in parsed["assessment"]
                else "assess-watch" if "PAST" in parsed["assessment"]
                else "assess-good")
        st.markdown(
            f'<div class="assess {acls}"><b>Assessment:</b> {parsed["assessment"]}</div>',
            unsafe_allow_html=True,
        )

        # Recommendations
        st.markdown("**Recommendations**")
        for rec in recommendations_for(parsed):
            st.markdown(f"- {rec}")

        # Full text kept available but tucked away
        with st.expander("View original report text"):
            st.text(st.session_state.doc_text)

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
