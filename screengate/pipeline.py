"""The gateway pipeline.

    request -> L1 patterns -> L4 semantic -> L2 spotlight -> model -> L3 outbound

Ordering is deliberate. The two detection layers run before the model so a
blocking verdict costs nothing - no token spend, no latency, and crucially
no chance of the model acting on the payload. Spotlighting runs last among
the inbound layers because it rewrites content, and the detectors should
see the original text, not the fenced version.

Blocking happens at two points, and the second is the important one: even
when the input looked clean, the output must still pass policy. Input
filtering alone fails against any phrasing the rules did not anticipate,
which is most of them.
"""

from __future__ import annotations

import time
import uuid
from typing import Callable

from .config import Policy, Settings
from .defenses import (
    InboundPatternDefense,
    OutboundPolicyDefense,
    SemanticDefense,
    SpotlightDefense,
)
from .defenses.outbound_policy import parse_result
from .models import (
    AuditRecord,
    Decision,
    Finding,
    GatewayRequest,
    GatewayResponse,
    max_severity,
)
from .providers import SYSTEM_PROMPT, Provider, build_provider
from .rag.embeddings import Embedder, build_embedder

BLOCK_MESSAGE = (
    "This request was blocked by policy. If you believe this is an error, "
    "quote the request ID to your administrator."
)


class Gateway:
    def __init__(
        self,
        provider: Provider | None = None,
        policy: Policy | None = None,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        audit_sink: Callable[[AuditRecord], None] | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.provider = provider or build_provider(self.settings)
        self.policy = policy or Policy()
        self.audit_sink = audit_sink

        self._patterns = InboundPatternDefense()
        self._spotlight = SpotlightDefense()
        self._outbound = OutboundPolicyDefense()
        self._semantic = SemanticDefense(embedder or build_embedder(self.settings))

    @property
    def semantic_reportable(self) -> bool:
        return self._semantic.reportable

    def handle(
        self,
        request: GatewayRequest,
        policy: Policy | None = None,
        prior_findings: list[Finding] | None = None,
    ) -> GatewayResponse:
        """Screen one request.

        `prior_findings` carries detections made before the text reached
        the gateway - currently PDF concealment, which can only be seen
        while the original file is in hand. They participate in the
        blocking decision exactly like any other finding, so a resume with
        text hidden under white fill is stopped even when the words
        themselves look innocuous.
        """
        policy = policy or self.policy
        started = time.perf_counter()
        request_id = uuid.uuid4().hex[:12]

        segments = request.segments()
        original = list(segments)
        findings: list[Finding] = list(prior_findings or [])
        layers_run: list[str] = ["ingest"] if prior_findings else []

        # Fail closed before any layer runs: a document whose text was
        # hidden from the reader is already disqualifying.
        if policy.decision_for(max_severity(findings)) is Decision.BLOCK:
            return self._finish(
                request, request_id, Decision.BLOCK, None, None,
                findings, layers_run, started,
            )

        # ---- inbound detection ----------------------------------------
        if policy.layers.inbound_patterns:
            res = self._patterns.inspect(segments, policy)
            findings.extend(res.findings)
            segments = res.segments
            original = list(segments)
            layers_run.append(self._patterns.name)

        if policy.layers.semantic:
            res = self._semantic.inspect(segments, policy)
            findings.extend(res.findings)
            layers_run.append(self._semantic.name)

        # Fail closed: never spend a model call on a request already blocking.
        if policy.decision_for(max_severity(findings)) is Decision.BLOCK:
            return self._finish(
                request, request_id, Decision.BLOCK, None, None,
                findings, layers_run, started,
            )

        # ---- structural rewrite ---------------------------------------
        if policy.layers.spotlighting:
            res = self._spotlight.inspect(segments, policy)
            findings.extend(res.findings)
            segments = res.segments
            layers_run.append(self._spotlight.name)

        # ---- model ----------------------------------------------------
        output = self.provider.complete(SYSTEM_PROMPT, segments)

        # ---- outbound -------------------------------------------------
        if policy.layers.outbound_policy:
            # Checked against the *original* segments: the fenced copies
            # contain the gateway's own preamble, which would pollute the
            # evidence-overlap calculation.
            findings.extend(self._outbound.inspect(output, original, policy))
            layers_run.append(self._outbound.name)

        decision = policy.decision_for(max_severity(findings))
        result, _ = parse_result(output)
        return self._finish(
            request, request_id, decision, output, result,
            findings, layers_run, started,
        )

    # ---- internals ----------------------------------------------------

    def _finish(
        self,
        request: GatewayRequest,
        request_id: str,
        decision: Decision,
        output: str | None,
        result,
        findings: list[Finding],
        layers_run: list[str],
        started: float,
    ) -> GatewayResponse:
        latency_ms = (time.perf_counter() - started) * 1000
        blocked = decision is Decision.BLOCK

        if self.audit_sink is not None:
            self.audit_sink(
                AuditRecord(
                    request_id=request_id,
                    session_id=request.session_id,
                    decision=decision,
                    findings=[f.redacted() for f in findings],
                    layers_run=layers_run,
                    latency_ms=latency_ms,
                    input_chars=sum(len(s.content) for s in request.segments()),
                )
            )

        return GatewayResponse(
            decision=decision,
            output=BLOCK_MESSAGE if blocked else output,
            result=None if blocked else result,
            findings=findings,
            layers_run=layers_run,
            latency_ms=latency_ms,
            request_id=request_id,
        )
