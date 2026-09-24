# 🩺 DengueCare RAG Assistant

**Project 1 — Multimodal Dengue Report RAG Assistant**

A healthcare-information demonstration assistant. A user selects/uploads a **synthetic**
dengue test report and asks a natural-language question. The system retrieves relevant
information from the report (Retrieval-Augmented Generation) and uses **Amazon Bedrock**
to generate a grounded answer, returning the answer **plus the evidence** it used.

> Uses synthetic medical data only. Not medical advice.

---

## What it demonstrates

- A web-based RAG chatbot with a simple UI.
- Synthetic dengue reports stored in **Amazon S3**.
- **Amazon Bedrock Knowledge Base** connected to the S3 document repository.
- **Vector retrieval** (OpenSearch Serverless) behind the scenes.
- **Bedrock foundation model** generating grounded answers.
- API-based UI/backend integration (**Lambda + API Gateway**, or FastAPI).
- Privacy, safety, and monitoring controls (**PII redaction / Guardrails / CloudWatch**).

---

## Architecture

```
User / Web UI
      │  (question [+ report])
      ▼
API Gateway ──► Lambda ──► Amazon Bedrock Knowledge Base
                                 │        │
                        Vector Retrieval  │ (OpenSearch Serverless
                        (embeddings)       │  vectors from S3 reports)
                                 ▼        ▼
                        Bedrock Foundation Model (Claude)
                                 │
                        Guardrails / validation
                                 ▼
                        API response (answer + evidence)
                                 ▼
                            Web UI
```

S3 stores the reports; the Knowledge Base embeds + indexes them; a question triggers
vector search, the retrieved chunks are sent to the foundation model, and a grounded
answer + citations return to the UI.

---

## Three run modes (set `RAG_MODE` in `.env`)

| Mode    | What it does                                                        | Needs AWS? |
|---------|---------------------------------------------------------------------|------------|
| `mock`  | Deterministic demo answers, in-code retrieval. Guaranteed to run.   | No         |
| `local` | Bedrock embeddings + in-code vector search + Bedrock generation.    | Bedrock    |
| `kb`    | **Real managed RAG** via Bedrock Knowledge Base (`retrieve_and_generate`). | Full AWS |

This lets you demo instantly (`mock`), then flip to the real AWS path (`kb`) once your
Knowledge Base is live — same UI, same API contract.

---

## Run locally (any mode)

```powershell
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Configure
Copy-Item .env.example .env        # defaults to RAG_MODE=mock

# 3. Start
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000

# 4. Open the UI
#    http://127.0.0.1:8000
```

Pick a bundled synthetic report (or upload your own `.txt` / `.pdf`), type a question
like *"What is the platelet count?"*, and click **Ask**.

To use real Bedrock without a Knowledge Base, set `RAG_MODE=local` and ensure Bedrock
model access is enabled. To use the full managed RAG path, set `RAG_MODE=kb` and
`KB_ID=<your-kb-id>`.

---

## Deploy to AWS

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for the full step-by-step guide:
S3 bucket, Knowledge Base creation, Lambda, API Gateway, IAM, CloudWatch, and Guardrails.

---

## AWS service mapping

| Service                       | Role              | Why it's used                                   |
|-------------------------------|-------------------|-------------------------------------------------|
| Amazon S3                     | Document storage  | Stores synthetic reports and uploads reliably.  |
| Amazon Bedrock                | Foundation model  | Generates natural-language answers from context.|
| Bedrock Knowledge Bases       | Managed RAG       | Connects docs → embeddings → retrieval → model.  |
| OpenSearch Serverless         | Vector storage    | Stores/searches report embeddings.              |
| AWS Lambda                    | Serverless backend| Runs the request/RAG logic without servers.     |
| API Gateway                   | API layer         | HTTP endpoint between UI and backend.           |
| IAM                           | Security          | Least-privilege access to S3/Bedrock/logs.      |
| CloudWatch                    | Monitoring        | Logs, errors, latency.                          |
| Bedrock Guardrails            | Safety            | Content + PII safety controls (optional).       |

---

## Responsible AI checklist

- [x] Synthetic/sample medical data only — no real patient data.
- [x] PII detection + redaction before the model call (`backend/pii.py`), and/or
      Bedrock Guardrails in KB mode.
- [x] Grounded answers: the model is instructed to answer only from retrieved context.
- [x] Evidence returned so answers are auditable.
- [x] Least-privilege IAM policies; no AWS keys in code (`aws configure` / IAM roles).
- [x] CloudWatch logging; raw sensitive content is not logged.
- [x] "Not medical advice" disclaimer shown in the UI.

---

## Project structure

```
project/
├── backend/            FastAPI app (API layer + 3 run modes)
│   ├── app.py          Endpoints: /health /reports /select-sample /upload /ask
│   ├── config.py       Env-driven configuration
│   ├── rag.py          Chunking + in-memory vector store + retrieval
│   ├── bedrock_client.py  Embeddings + generation (Bedrock, with mock fallback)
│   ├── kb_client.py    Bedrock Knowledge Base retrieve_and_generate (real AWS RAG)
│   └── pii.py          PII redaction (Responsible AI)
├── frontend/           Web UI (HTML/CSS/JS)
├── sample_reports/     Synthetic dengue reports
├── aws/                Lambda handler, IAM policies, S3 upload script
├── DEPLOYMENT.md       Full AWS deployment guide
├── requirements.txt
└── .env.example
```

## Test scenarios

| Report                          | Question                          | Expected            |
|---------------------------------|-----------------------------------|---------------------|
| synthetic_dengue_report_01.txt  | What is the platelet count?       | 95,000 /uL (low)    |
| synthetic_dengue_report_01.txt  | Is the NS1 antigen positive?      | Yes, positive       |
| synthetic_dengue_report_02.txt  | Is there an active dengue infection? | No; past exposure |
| synthetic_dengue_report_02.txt  | What is the platelet count?       | 210,000 /uL (normal)|
