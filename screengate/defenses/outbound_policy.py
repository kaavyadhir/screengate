"""Layer 3 - outbound policy enforcement.

The last line, and the only one that catches attacks nobody anticipated.
Layers 1, 2 and 4 all reason about the input. This one asks a different
and far more answerable question: *whatever was asked, is this output
allowed to leave?*

The backbone is the Pydantic contract. `ScreeningResult` is strict - score
bounded 0-100, verdict from a fixed set, between one and five pieces of
evidence, no extra fields. A model doing its job satisfies that trivially.
A hijacked model does not: it emits prose, or the system prompt, or a
score with no evidence. So `ValidationError` is read as a detection, not
swallowed as a parsing nuisance.

This is why the layer generalises. It never needs to know what the attack
was - only that the answer stopped looking like an answer.

Checks, in increasing order of generality:
  canary_leak          - the system prompt came back out
  data_exfiltration    - a link to a host not on the allowlist
  injection_compliance - the model says it is following new instructions
  contract_violation   - output does not satisfy ScreeningResult
  score_anomaly        - contract-valid, but the score is unsupported by
                         anything actually present in the resume
"""

from __future__ import annotations

import json
import re
from urllib.parse import urlparse

from pydantic import ValidationError

from ..config import Policy
from ..models import Finding, ScreeningResult, Segment, Severity, Trust

URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+", re.I)
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(\s*(\S+?)\s*\)")
JSON_BLOCK_RE = re.compile(r"\{.*\}", re.S)

COMPLIANCE_RE = re.compile(
    r"\b(?:disregarding|ignoring|overriding|setting aside)\b[\s\w]{0,30}"
    r"\b(?:instruction|rule|guideline|direction)s?\b"
    r"|\bhere are my (?:instructions|rules|system)\b"
    r"|\bas (?:requested|instructed)[,:]?\s*(?:here|i)\b",
    re.I,
)

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "on",
    "is", "are", "was", "were", "this", "that", "candidate", "resume",
    "experience", "years", "role", "match", "strong", "excellent",
}


def parse_result(output: str) -> tuple[ScreeningResult | None, str | None]:
    """Try to read the output as a ScreeningResult.

    Returns (result, error). A non-None error is a security signal.
    """
    text = (output or "").strip()
    if not text:
        return None, "empty output"

    block = JSON_BLOCK_RE.search(text)
    if not block:
        return None, "output is not JSON"

    try:
        data = json.loads(block.group(0))
    except json.JSONDecodeError as exc:
        return None, f"malformed JSON: {exc.msg}"

    if not isinstance(data, dict):
        return None, "JSON is not an object"

    try:
        return ScreeningResult.model_validate(data), None
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first.get("loc", ())) or "<root>"
        return None, f"{loc}: {first.get('msg', 'invalid')}"


class OutboundPolicyDefense:
    name = "outbound_policy"

    def inspect(
        self, output: str, segments: list[Segment], policy: Policy
    ) -> list[Finding]:
        findings: list[Finding] = []
        findings.extend(self._canaries(output, policy))
        findings.extend(self._links(output, policy))
        findings.extend(self._compliance(output))

        result, error = parse_result(output)
        if error is not None:
            findings.append(
                Finding(
                    layer=self.name,
                    rule="contract_violation",
                    severity=Severity.HIGH,
                    trust=Trust.MODEL,
                    message=(
                        f"Output does not satisfy the screening contract ({error}). "
                        "A model following its instructions returns valid "
                        "ScreeningResult JSON."
                    ),
                )
            )
        else:
            findings.extend(self._score_anomaly(result, segments, policy))

        return findings

    # ---- checks -------------------------------------------------------

    def _canaries(self, output: str, policy: Policy) -> list[Finding]:
        low = (output or "").lower()
        hits = [c for c in policy.system_prompt_canaries if c.lower() in low]
        if not hits:
            return []
        return [
            Finding(
                layer=self.name,
                rule="canary_leak",
                severity=Severity.CRITICAL,
                trust=Trust.MODEL,
                message=(
                    f"Output contains {len(hits)} system-prompt canary value(s); "
                    "operator instructions have leaked."
                ),
            )
        ]

    def _links(self, output: str, policy: Policy) -> list[Finding]:
        candidates = set(URL_RE.findall(output or "")) | set(
            MARKDOWN_IMAGE_RE.findall(output or "")
        )
        findings: list[Finding] = []
        for raw in candidates:
            host = (urlparse(raw).hostname or "").lower()
            if not host or self._allowed(host, policy):
                continue
            findings.append(
                Finding(
                    layer=self.name,
                    rule="data_exfiltration",
                    severity=Severity.CRITICAL,
                    trust=Trust.MODEL,
                    message=f"Output links to non-allowlisted host '{host}'.",
                    excerpt=raw,
                ).redacted()
            )
        return findings

    def _allowed(self, host: str, policy: Policy) -> bool:
        for allowed in policy.allowed_link_hosts:
            allowed = allowed.lower()
            if host == allowed or host.endswith("." + allowed):
                return True
        return False

    def _compliance(self, output: str) -> list[Finding]:
        m = COMPLIANCE_RE.search(output or "")
        if not m:
            return []
        return [
            Finding(
                layer=self.name,
                rule="injection_compliance",
                severity=Severity.CRITICAL,
                trust=Trust.MODEL,
                message="Output acknowledges following injected instructions.",
                excerpt=m.group(0),
            ).redacted()
        ]

    def _score_anomaly(
        self, result: ScreeningResult, segments: list[Segment], policy: Policy
    ) -> list[Finding]:
        """Catch contract-valid output whose score nothing supports.

        An injection that successfully dictates a score usually also
        supplies thin, generic evidence - because the attacker is writing
        the conclusion, not reading the resume. So a very high score whose
        evidence shares almost no vocabulary with the retrieved resume text
        is treated as unsupported.

        Deliberately conservative: it only fires at or above
        score_anomaly_floor, so ordinary strong candidates are unaffected.
        """
        if result.score < policy.score_anomaly_floor:
            return []

        doc_text = " ".join(
            s.content for s in segments if s.trust is Trust.DOCUMENT
        ).lower()
        if not doc_text.strip():
            return [
                Finding(
                    layer=self.name,
                    rule="score_anomaly",
                    severity=Severity.HIGH,
                    trust=Trust.MODEL,
                    message=(
                        f"Score of {result.score} awarded with no resume text "
                        "available to support it."
                    ),
                    score=float(result.score),
                )
            ]

        doc_tokens = set(re.findall(r"[a-z]{3,}", doc_text)) - STOPWORDS
        ev_tokens = set(
            re.findall(r"[a-z]{3,}", " ".join(result.evidence).lower())
        ) - STOPWORDS

        if not ev_tokens:
            overlap = 0.0
        else:
            overlap = len(ev_tokens & doc_tokens) / len(ev_tokens)

        if overlap >= 0.30:
            return []

        return [
            Finding(
                layer=self.name,
                rule="score_anomaly",
                severity=Severity.HIGH,
                trust=Trust.MODEL,
                message=(
                    f"Score of {result.score} is supported by evidence that "
                    f"overlaps the resume text by only {overlap:.0%}; the "
                    "conclusion is not grounded in the document."
                ),
                score=round(overlap, 3),
            )
        ]
