"""Defense layer interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..config import Policy
from ..models import Finding, Segment


@dataclass
class InboundResult:
    findings: list[Finding]
    segments: list[Segment]
    """Possibly rewritten segments - spotlighting edits content in place."""


class InboundDefense(Protocol):
    name: str

    def inspect(self, segments: list[Segment], policy: Policy) -> InboundResult: ...


class OutboundDefense(Protocol):
    name: str

    def inspect(
        self, output: str, segments: list[Segment], policy: Policy
    ) -> list[Finding]: ...
