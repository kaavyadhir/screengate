"""Core data models.

Two ideas drive everything in this file.

1. **Provenance is a first-class property.** Every piece of content carries
   where it came from. Content the operator wrote is trusted; content a
   candidate uploaded is not, and content that arrived through retrieval is
   not instructions at all. Most prompt-injection defenses fail because the
   model sees one flat string with no idea which part is whose.

2. **The output has a contract.** `ScreeningResult` is not just response
   formatting - it is a security control. A hijacked model stops producing
   output that fits this shape, so a validation failure is a detection
   signal rather than an error to swallow. See defenses/outbound_policy.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Trust(str, Enum):
    """Where a piece of content came from.

    OPERATOR  - written by whoever deployed the app: system prompt, the job
                description, policy. Trusted.
    USER      - typed by the recruiter using the tool. Untrusted, but
                expected to be imperative - recruiters give instructions.
    DOCUMENT  - candidate resumes, retrieved chunks, anything uploaded.
                Untrusted AND has no business being imperative. This is the
                indirect-injection vector.
    MODEL     - produced by the LLM. Untrusted on the way out.
    """

    OPERATOR = "operator"
    USER = "user"
    DOCUMENT = "document"
    MODEL = "model"


class Decision(str, Enum):
    ALLOW = "allow"
    FLAG = "flag"
    BLOCK = "block"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


SEVERITY_ORDER: dict[Severity, int] = {
    Severity.LOW: 0,
    Severity.MEDIUM: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


class Finding(BaseModel):
    """One detection from one defense layer."""

    layer: str
    rule: str = Field(description="Stable identifier, used for metrics.")
    severity: Severity
    trust: Trust = Field(description="Trust zone the offending content came from.")
    message: str
    excerpt: str | None = Field(
        default=None, description="Short redacted excerpt. Never the full payload."
    )
    score: float | None = Field(
        default=None, description="Similarity score, for the semantic layer."
    )
    operational: bool = Field(
        default=False,
        description=(
            "True when this reports the state of the gateway rather than "
            "something about the content - a layer that could not run, for "
            "example. Operational findings are reported but do not affect "
            "the verdict: a candidate must not be flagged because a quota "
            "expired."
        ),
    )

    def redacted(self, limit: int = 120) -> "Finding":
        if self.excerpt is None:
            return self
        text = " ".join(self.excerpt.split())
        if len(text) > limit:
            text = text[:limit] + "..."
        return self.model_copy(update={"excerpt": text})


class Segment(BaseModel):
    """Content plus its provenance."""

    trust: Trust
    content: str
    label: str | None = Field(
        default=None, description="Source name, e.g. 'arjun_mehta_resume.pdf'."
    )


# --------------------------------------------------------------------------
# The output contract. This is a security control, not a response format.
# --------------------------------------------------------------------------


class ScreeningResult(BaseModel):
    """What the screening app is contractually required to return.

    The constraints are deliberately tight. A model following its
    instructions produces this easily; a model that has been hijacked
    produces prose, or a score outside the range, or no evidence at all.
    Every one of those is a ValidationError, which layer 3 reads as a
    detection rather than a bug.
    """

    model_config = {"extra": "forbid"}

    score: int = Field(ge=0, le=100, description="Match score against the role.")
    verdict: Literal["advance", "reject", "review"]
    evidence: list[str] = Field(
        min_length=1,
        max_length=5,
        description="Specific facts from the resume supporting the score.",
    )

    @property
    def is_strong(self) -> bool:
        return self.score >= 85


class GatewayRequest(BaseModel):
    """What a client sends in."""

    user_input: str = Field(description="The recruiter's instruction or question.")
    documents: list[Segment] = Field(default_factory=list)
    job_description: str | None = None
    session_id: str | None = None

    def segments(self) -> list[Segment]:
        return [Segment(trust=Trust.USER, content=self.user_input), *self.documents]


class GatewayResponse(BaseModel):
    decision: Decision
    output: str | None = Field(default=None, description="Raw model output, or None if blocked.")
    result: ScreeningResult | None = Field(
        default=None, description="Parsed result, when the output satisfied the contract."
    )
    findings: list[Finding] = Field(default_factory=list)
    layers_run: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    request_id: str = ""

    @property
    def blocked(self) -> bool:
        return self.decision is Decision.BLOCK


class AuditRecord(BaseModel):
    """One line in the audit log.

    Stores verdicts and rule names, never full payloads. A security log
    that mirrors every request is a second copy of the data you were
    trying to protect.
    """

    request_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str | None = None
    decision: Decision
    findings: list[Finding] = Field(default_factory=list)
    layers_run: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    input_chars: int = 0

    def to_log_line(self) -> dict[str, Any]:
        sev = max_severity(self.findings)
        return {
            "request_id": self.request_id,
            "ts": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "decision": self.decision.value,
            "rules": [f.rule for f in self.findings],
            "max_severity": sev.value if sev else None,
            "layers": self.layers_run,
            "latency_ms": round(self.latency_ms, 2),
            "input_chars": self.input_chars,
        }


def max_severity(findings: list[Finding]) -> Severity | None:
    """Highest severity among findings ABOUT THE CONTENT.

    Operational findings are excluded deliberately. A layer that cannot
    run - an expired embedding quota, say - is a fact about the gateway,
    not about the candidate, and letting it raise the severity would
    mark every single applicant for human review because a billing limit
    was reached. The finding still travels in the response so the
    degradation stays visible; it simply does not vote on the verdict.
    """
    relevant = [f for f in findings if not f.operational]
    if not relevant:
        return None
    return max((f.severity for f in relevant), key=lambda s: SEVERITY_ORDER[s])
