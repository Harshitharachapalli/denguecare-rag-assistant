"""AWS Lambda handler for the DengueCare RAG Assistant.

Deployed behind API Gateway. Receives a question, queries the Amazon
Bedrock Knowledge Base with retrieve_and_generate (managed RAG: embed ->
vector search -> foundation model), optionally applies a Guardrail, and
returns a grounded answer plus citations (evidence).

Environment variables (set on the Lambda):
  KB_ID              (required)  Bedrock Knowledge Base ID
  AWS_REGION         (provided by Lambda automatically)
  TEXT_MODEL_ID      (required)  e.g. anthropic.claude-3-haiku-20240307-v1:0
  GUARDRAIL_ID       (optional)
  GUARDRAIL_VERSION  (optional, default DRAFT)

Request (POST body JSON): {"question": "What is the platelet count?"}
Response JSON:            {"answer": "...", "evidence": [{"text","source"}]}
"""
import json
import os

import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
KB_ID = os.environ["KB_ID"]
TEXT_MODEL_ID = os.environ.get(
    "TEXT_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"
)
GUARDRAIL_ID = os.environ.get("GUARDRAIL_ID", "").strip()
GUARDRAIL_VERSION = os.environ.get("GUARDRAIL_VERSION", "DRAFT").strip()

_client = boto3.client("bedrock-agent-runtime", region_name=REGION)

_CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "OPTIONS,POST",
    "Content-Type": "application/json",
}


def _response(status, body):
    return {"statusCode": status, "headers": _CORS, "body": json.dumps(body)}


def handler(event, context):
    # CORS preflight
    if event.get("httpMethod") == "OPTIONS":
        return _response(200, {"ok": True})

    # Parse question from body (API Gateway proxy integration)
    try:
        raw = event.get("body") or "{}"
        payload = json.loads(raw) if isinstance(raw, str) else raw
        question = (payload.get("question") or "").strip()
    except (ValueError, AttributeError):
        return _response(400, {"error": "Invalid JSON body."})

    if not question:
        return _response(400, {"error": "Field 'question' is required."})

    model_arn = f"arn:aws:bedrock:{REGION}::foundation-model/{TEXT_MODEL_ID}"

    kb_config = {
        "knowledgeBaseId": KB_ID,
        "modelArn": model_arn,
        "retrievalConfiguration": {
            "vectorSearchConfiguration": {"numberOfResults": 3}
        },
    }
    if GUARDRAIL_ID:
        kb_config["generationConfiguration"] = {
            "guardrailConfiguration": {
                "guardrailId": GUARDRAIL_ID,
                "guardrailVersion": GUARDRAIL_VERSION,
            }
        }

    try:
        resp = _client.retrieve_and_generate(
            input={"text": question},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": kb_config,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return _response(502, {"error": f"Bedrock error: {exc}"})

    answer = resp.get("output", {}).get("text", "")
    evidence = []
    for citation in resp.get("citations", []):
        for ref in citation.get("retrievedReferences", []):
            evidence.append(
                {
                    "text": ref.get("content", {}).get("text", ""),
                    "source": ref.get("location", {})
                    .get("s3Location", {})
                    .get("uri", "knowledge-base"),
                }
            )

    return _response(200, {"answer": answer, "evidence": evidence})
