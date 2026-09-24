"""PII detection and redaction (Responsible AI control).

Runs BEFORE text is sent to the foundation model so that unnecessary
sensitive data (phone numbers, emails, national IDs, account numbers)
is masked. This is a lightweight stand-in for Amazon Bedrock Guardrails.
"""
import re

# Order matters: more specific patterns first.
_PATTERNS = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("PHONE", re.compile(r"\b(?:\+?\d{1,3}[-\s]?)?\d{10}\b")),
    # Aadhaar-like 12 digit number (India national ID) grouped or plain
    ("NATIONAL_ID", re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")),
    # Generic long account numbers (8-16 digits)
    ("ACCOUNT_NO", re.compile(r"\b\d{8,16}\b")),
]


def redact(text: str) -> tuple[str, list[str]]:
    """Return (redacted_text, list_of_pii_types_found)."""
    found: list[str] = []
    redacted = text
    for label, pattern in _PATTERNS:
        if pattern.search(redacted):
            found.append(label)
            redacted = pattern.sub(f"[REDACTED_{label}]", redacted)
    return redacted, sorted(set(found))


def contains_pii(text: str) -> bool:
    _, found = redact(text)
    return len(found) > 0
