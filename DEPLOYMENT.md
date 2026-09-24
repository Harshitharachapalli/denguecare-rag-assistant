# DengueCare RAG Assistant — AWS Deployment Guide

This guide takes you from zero to a working AWS RAG assistant:
**S3 → Bedrock Knowledge Base (vector store) → Lambda → API Gateway → Web UI**, with
IAM, CloudWatch, and an optional Bedrock Guardrail.

> Tip for a half-day timeline: run the app in **mock mode** first (see README) so you
> always have a working demo, then follow these steps to bring the real AWS path online.

---

## 0. Prerequisites

- An AWS account with permission to use Bedrock, S3, Lambda, API Gateway, IAM.
- AWS CLI configured: `aws configure` (region e.g. `us-east-1`).
- **Bedrock model access enabled** (Bedrock console → *Model access* → enable
  **Anthropic Claude 3 Haiku** and **Titan Text Embeddings V2**). This can take a few minutes.

Pick one region and use it everywhere. Examples below use `us-east-1`.

---

## 1. Create the S3 bucket and upload reports

```powershell
aws s3 mb s3://denguecare-reports-demo --region us-east-1
python aws/upload_reports_to_s3.py denguecare-reports-demo reports/
```

S3 is the durable document repository (the "knowledge source").

---

## 2. Create the Bedrock Knowledge Base (managed RAG)

Console path: **Amazon Bedrock → Knowledge Bases → Create knowledge base**.

1. **Name**: `denguecare-kb`.
2. **IAM role**: let Bedrock create a service role (or create one using
   `aws/kb-service-role-policy.json` — replace the bucket name first).
3. **Data source**: choose **Amazon S3**, point it at
   `s3://denguecare-reports-demo/reports/`.
4. **Embeddings model**: **Titan Text Embeddings V2**.
5. **Vector store**: choose **Quick create** — Bedrock provisions
   **Amazon OpenSearch Serverless** for you (this is the vector storage that holds
   the report embeddings). *(Provisioning takes ~10–15 min.)*
6. Create the KB, then open the **data source** and click **Sync**. Syncing:
   - reads the reports from S3,
   - chunks them,
   - creates embeddings with Titan,
   - stores vectors in OpenSearch Serverless.
7. Copy the **Knowledge Base ID** (looks like `XXXXXXXXXX`). You need it next.

Test it right in the console with a question like *"What is the platelet count?"*.

---

## 3. Deploy the Lambda function

The handler is `aws/lambda_function.py` (`handler` entry point).

**Create the execution role** with the policy in `aws/lambda-iam-policy.json`
(Bedrock RetrieveAndGenerate + InvokeModel + CloudWatch Logs), trusted by
`lambda.amazonaws.com`.

Package and create the function:

```powershell
# From the project root
Compress-Archive -Path aws\lambda_function.py -DestinationPath lambda.zip -Force

aws lambda create-function `
  --function-name denguecare-rag `
  --runtime python3.12 `
  --handler lambda_function.handler `
  --zip-file fileb://lambda.zip `
  --role arn:aws:iam::<ACCOUNT_ID>:role/denguecare-lambda-role `
  --timeout 30 `
  --environment "Variables={KB_ID=<YOUR_KB_ID>,TEXT_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0}" `
  --region us-east-1
```

boto3 is included in the Lambda Python runtime, so no extra packaging is needed.

To update code later:

```powershell
Compress-Archive -Path aws\lambda_function.py -DestinationPath lambda.zip -Force
aws lambda update-function-code --function-name denguecare-rag --zip-file fileb://lambda.zip --region us-east-1
```

---

## 4. Expose it with API Gateway (HTTP API)

1. **API Gateway → Create API → HTTP API**.
2. Add integration: **Lambda** → `denguecare-rag`.
3. Add route: `POST /ask` → the Lambda integration.
4. Enable **CORS** (allow origin `*`, method `POST`, header `Content-Type`).
5. Deploy and copy the **Invoke URL** (e.g. `https://abc123.execute-api.us-east-1.amazonaws.com`).

Test the endpoint:

```powershell
curl -X POST "https://abc123.execute-api.us-east-1.amazonaws.com/ask" `
  -H "Content-Type: application/json" `
  -d '{"question":"What is the platelet count?"}'
```

You should get `{"answer": "...", "evidence": [...]}`.

---

## 5. Point the Web UI at the API

The frontend calls same-origin `/ask` by default (local mode). To call API Gateway
instead, edit `frontend/app.js` and set the fetch URLs to your Invoke URL, then host
the `frontend/` folder as a static site (e.g. an S3 static website or any static host).

Alternatively, keep the FastAPI backend (`RAG_MODE=kb`, `KB_ID=<id>`) as your API
layer — it calls the same Knowledge Base. This is the "optional FastAPI architecture"
from the spec and is the simplest way to demo the full path.

---

## 6. (Optional) Add a Bedrock Guardrail

1. **Bedrock → Guardrails → Create guardrail**. Add content filters and a
   sensitive-information (PII) policy.
2. Copy the **Guardrail ID** and version.
3. Set `GUARDRAIL_ID` (and `GUARDRAIL_VERSION`) on the Lambda env vars (or in `.env`
   for FastAPI). The code automatically attaches it to the RAG request.

---

## 7. Monitoring with CloudWatch

- Lambda automatically logs to CloudWatch Logs (`/aws/lambda/denguecare-rag`).
- View invocations, errors, and latency under **CloudWatch → Log groups** and
  **Lambda → Monitor**.
- The IAM policy already grants the required `logs:*` permissions.

---

## 8. Teardown (avoid charges)

- Delete the API Gateway API.
- Delete the Lambda function and its role.
- Delete the Knowledge Base **and** the OpenSearch Serverless collection it created.
- Empty and delete the S3 bucket.
- Delete the Guardrail if created.

OpenSearch Serverless has an hourly cost, so remember to delete it after the demo.
