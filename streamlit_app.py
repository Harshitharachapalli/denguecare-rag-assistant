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

    # Patterns tolerate both "Field : value" and "Field    value" layouts.
    meta = {
        "Report No.": find(r"Report No\.?\s*[:]?\s*(\S+)"),
        "Patient": find(r"Patient Name\s*[:]?\s*(.+?)\s{2,}"),
        "Age / Sex": find(r"Age / Sex\s*[:]?\s*(.+?)\s{2,}"),
        "Collected": find(r"Collected On\s*[:]?\s*(.+?)\s{2,}"),
    }
    meta = {k: v for k, v in meta.items() if v}

    rows = []

    # --- Serology / antigen: qualitative Positive/Negative ---
    serology = {
        "NS1 Antigen": find(r"NS1 Antigen\s*[:]?\s+(Positive|Negative|Reactive|Non-reactive)"),
        "IgM Antibody": find(r"IgM Antibody\s*[:]?\s+(Positive|Negative|Reactive|Non-reactive)"),
        "IgG Antibody": find(r"IgG Antibody\s*[:]?\s+(Positive|Negative|Reactive|Non-reactive)"),
    }
    for test, val in serology.items():
        if not val:
            continue
        positive = val.upper() in ("POSITIVE", "REACTIVE")
        if positive:
            # NS1 / IgM positive => active concern (Alert); IgG positive => past exposure (Watch)
            status = "Positive"
            verdict = "Watch" if test == "IgG Antibody" else "Alert"
        else:
            status, verdict = "Negative", "Good"
        rows.append({"test": test, "value": status, "range": "Negative",
                     "status": status, "verdict": verdict})

    # --- Numeric CBC values with ranges ---
    plt = num(find(r"Platelet Count\s*[:]?\s+([\d,]+)"))
    if plt is not None:
        if plt < 150000:
            status, verdict = "Low", "Alert"
        elif plt > 410000:
            status, verdict = "High", "Watch"
        else:
            status, verdict = "Normal", "Good"
        rows.append({"test": "Platelet Count", "value": f"{int(plt):,} /uL",
                     "range": "150,000 - 410,000", "status": status, "verdict": verdict})

    wbc = num(find(r"Total WBC Count\s*[:]?\s+([\d,]+)"))
    if wbc is not None:
        if wbc < 4000:
            status, verdict = "Low", "Watch"
        elif wbc > 11000:
            status, verdict = "High", "Watch"
        else:
            status, verdict = "Normal", "Good"
        rows.append({"test": "Total WBC Count", "value": f"{int(wbc):,} /uL",
                     "range": "4,000 - 11,000", "status": status, "verdict": verdict})

    hb = num(find(r"Hemoglobin\s*[:]?\s+([\d.]+)"))
    if hb is not None:
        if hb < 12.0:
            status, verdict = "Low", "Watch"
        elif hb > 16.0:
            status, verdict = "High", "Watch"
        else:
            status, verdict = "Normal", "Good"
        rows.append({"test": "Hemoglobin", "value": f"{hb} g/dL",
                     "range": "12.0 - 16.0", "status": status, "verdict": verdict})

    hct = num(find(r"Hematocrit \(PCV\)\s*[:]?\s+(\d+)"))
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


# Clean, light, high-contrast styling with subtle motion.
st.markdown(
    """
    <style>
      /* ---------- App background: soft light gradient ---------- */
      .stApp {
        background:
          radial-gradient(900px 500px at 8% -8%, #e0ecff 0%, transparent 55%),
          radial-gradient(800px 460px at 108% 0%, #d8f5ee 0%, transparent 50%),
          linear-gradient(180deg, #f4f7fc 0%, #eef2f9 100%);
        background-attachment: fixed;
        color: #1f2937;
      }
      .block-container { max-width: 1160px; padding-top: 1.4rem; }

      /* ---------- Animations ---------- */
      @keyframes rise { from {opacity:0; transform: translateY(12px);} to {opacity:1; transform:none;} }
      @keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(239,68,68,.35);}
                         70%{box-shadow:0 0 0 7px rgba(239,68,68,0);}
                         100%{box-shadow:0 0 0 0 rgba(239,68,68,0);} }

      /* ---------- Hero header ---------- */
      .hero {
        background: linear-gradient(120deg, #2563eb 0%, #0ea5e9 55%, #14b8a6 100%);
        border-radius: 18px; padding: 24px 30px; margin-bottom: 20px;
        box-shadow: 0 12px 30px rgba(37,99,235,.28);
        animation: rise .5s ease both;
      }
      .hero h1 { margin: 0; font-size: 1.95rem; font-weight: 800;
                 letter-spacing: -.5px; color: #ffffff; }
      .hero p { margin: 8px 0 0; color: rgba(255,255,255,.92); font-size: .98rem; }

      /* ---------- Cards (Streamlit columns) ---------- */
      div[data-testid="column"] > div {
        background: #ffffff;
        border: 1px solid #e6eaf2;
        border-radius: 16px; padding: 20px 22px;
        box-shadow: 0 6px 20px rgba(31,41,59,.06);
        animation: rise .5s ease both;
        transition: transform .16s ease, box-shadow .16s ease;
      }
      div[data-testid="column"] > div:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(31,41,59,.10);
      }

      h1, h2, h3, h4 { color: #0f172a; }
      .muted { color: #64748b !important; font-size: .9rem; }

      /* ---------- Answer box ---------- */
      .answer {
        background: #eff6ff; border: 1px solid #dbeafe; border-left: 4px solid #2563eb;
        border-radius: 12px; padding: 16px 18px; line-height: 1.6;
        color: #0f172a; white-space: pre-wrap; animation: rise .4s ease both;
      }
      .evi {
        border-left: 3px solid #cbd5e1; padding: 8px 12px; margin-bottom: 8px;
        color: #475569; font-size: .86rem; white-space: pre-wrap;
        background: #f8fafc; border-radius: 6px;
      }

      /* ---------- Results table ---------- */
      table.rpt { width: 100%; border-collapse: collapse; font-size: .9rem; margin: 8px 0 12px; }
      table.rpt th {
        text-align: left; padding: 10px 12px; font-weight: 700; color: #475569;
        background: #f1f5f9; border-bottom: 2px solid #e2e8f0;
        text-transform: uppercase; font-size: .72rem; letter-spacing: .05em;
      }
      table.rpt td { padding: 10px 12px; border-bottom: 1px solid #eef2f7; color: #1f2937; }
      table.rpt tbody tr { transition: background .15s ease; }
      table.rpt tbody tr:hover { background: #f5f9ff; }

      /* ---------- Status badges ---------- */
      .badge { display: inline-block; padding: 3px 12px; border-radius: 999px;
               font-size: .76rem; font-weight: 700; }
      .b-green  { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
      .b-red    { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
      .b-yellow { background: #fef9c3; color: #a16207; border: 1px solid #fde047; }
      .b-orange { background: #ffedd5; color: #c2410c; border: 1px solid #fdba74; }
      .b-grey   { background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }

      /* ---------- Assessment banner ---------- */
      .assess { border-radius: 12px; padding: 12px 16px; font-size: .94rem; margin: 10px 0 6px;
                animation: rise .5s ease both; }
      .assess-alert { background: #fef2f2; border: 1px solid #fecaca; color: #991b1b; }
      .assess-watch { background: #fffbeb; border: 1px solid #fde68a; color: #92400e; }
      .assess-good  { background: #f0fdf4; border: 1px solid #bbf7d0; color: #166534; }

      /* ---------- Buttons ---------- */
      .stButton > button {
        border-radius: 10px; font-weight: 700; border: 0; color: #fff !important;
        background: linear-gradient(120deg, #2563eb, #0ea5e9);
        transition: transform .08s ease, box-shadow .2s ease, filter .2s ease;
        box-shadow: 0 6px 16px rgba(37,99,235,.30);
      }
      .stButton > button:hover { transform: translateY(-2px); filter: brightness(1.05);
        box-shadow: 0 10px 22px rgba(14,165,233,.35); }
      .stButton > button:active { transform: translateY(0); }

      /* ---------- Sidebar: light and readable ---------- */
      section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e6eaf2;
      }
      section[data-testid="stSidebar"] * { color: #1f2937 !important; }
      section[data-testid="stSidebar"] .muted { color: #64748b !important; }

      #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Sidebar: choose a report.
with st.sidebar:
    st.markdown("## 🩺 Dengue Assistant")
    st.markdown('<span class="muted">Choose a report to begin.</span>',
                unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Select a report")

    if not config.USE_KB:
        samples = list_samples()
        picked = st.selectbox("Sample reports", ["Select a report..."] + samples)
        if st.button("Load report", use_container_width=True):
            if picked and picked != "Select a report...":
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
st.markdown(
    """
    <div class="hero">
      <h1>Dengue Report Intelligence</h1>
      <p>Load a dengue lab report, review the analyzed results, and ask
      natural-language questions answered from the report itself.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_main, col_report = st.columns([3, 2], gap="large")

# One fixed color per status word.
#   Positive / Normal -> green, Negative -> red, Low -> yellow, High -> orange
_BADGE_CLASS = {
    "Positive": "b-green",
    "Normal": "b-green",
    "Negative": "b-red",
    "Low": "b-yellow",
    "High": "b-orange",
}


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
                cls = _BADGE_CLASS.get(r["status"], "b-grey")
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
