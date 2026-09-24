"""Amazon Bedrock Knowledge Base client.

This is the REAL AWS RAG path. It calls the managed
`bedrock-agent-runtime.retrieve_and_generate` API, which internally:
  1. embeds the question,
  2. runs vector search over the Knowledge Base vector store,
  3. builds a grounded prompt,
  4. calls the Bedrock foundation model,
  5. returns the answer WITH citations (evidence).

Used when RAG_MODE=kb and KB_ID is set.
"""
import logging

from . import config

logger = logging.getLogger("kb")

_agent_rt = None


def _client():
    global _agent_rt
    if _agent_rt is None:
        import boto3

        _agent_rt = boto3.client(
            "bedrock-agent-runtime", region_name=config.AWS_REGION
        )
    return _agent_rt


def retrieve_and_generate(question: str) -> dict:
    """Query the Knowledge Base and return {answer, evidence[]}.

    evidence items: {"score": float|None, "text": str, "source": str}
    """
    model_arn = (
        f"arn:aws:bedrock:{config.AWS_REGION}::foundation-model/"
        f"{config.BEDROCK_TEXT_MODEL_ID}"
    )

    request = {
        "input": {"text": question},
        "retrieveAndGenerateConfiguration": {
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": config.KB_ID,
                "modelArn": model_arn,
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {
                        "numberOfResults": config.TOP_K
                    }
                },
            },
        },
    }

    # Optional Bedrock Guardrail
    if config.GUARDRAIL_ID:
        request["retrieveAndGenerateConfiguration"]["knowledgeBaseConfiguration"][
            "generationConfiguration"
        ] = {
            "guardrailConfiguration": {
                "guardrailId": config.GUARDRAIL_ID,
                "guardrailVersion": config.GUARDRAIL_VERSION,
            }
        }

    resp = _client().retrieve_and_generate(**request)

    answer = resp.get("output", {}).get("text", "")

    evidence = []
    for citation in resp.get("citations", []):
        for ref in citation.get("retrievedReferences", []):
            text = ref.get("content", {}).get("text", "")
            location = ref.get("location", {})
            source = (
                location.get("s3Location", {}).get("uri")
                or location.get("type", "")
                or "knowledge-base"
            )
            evidence.append({"score": None, "text": text, "source": source})

    return {"answer": answer, "evidence": evidence}
