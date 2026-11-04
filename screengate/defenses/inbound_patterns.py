"""Layer 1 - provenance-scoped pattern matching.

The central idea, and the reason this layer does not drown in false
positives: *the same words mean different things depending on who said
them.*

A recruiter typing "ignore what I said, use this resume instead" is making
an ordinary request. A resume containing "ignore your previous
instructions" is an attack, because a resume has no legitimate reason to
address the model in the imperative at all.

So every rule carries two severities - one for USER-zone content, one for
DOCUMENT-zone content - and it is the document-zone severity that blocks.
Pattern matching alone is a blunt instrument. Pattern matching scoped by
provenance is a usable one.

Known limitation, stated plainly because it shows up in the benchmark:
this layer cannot tell a document *quoting* an instruction from a document
*issuing* one. A resume describing "I was told to ignore the previous
instructions and rewrite the module" is legitimate text that trips the
override rule. Suppressing reported speech would fix the false positive
and immediately open an evasion path, so the false positive is kept and
reported. Layer 3 is the architectural answer: layer 1 can afford to be
conservative because it is not the last line.
"""

from __future__ import annotations

import re
import unicodedata

from ..config import Policy
from ..models import Finding, Segment, Severity, Trust
from .base import InboundResult


class Rule:
    def __init__(
        self,
        rule_id: str,
        pattern: str,
        *,
        doc_severity: Severity,
        user_severity: Severity | None,
        message: str,
        downgrade_pattern: str | None = None,
        downgraded_severity: Severity | None = None,
    ) -> None:
        self.rule_id = rule_id
        self.pattern = re.compile(pattern, re.I)
        self.doc_severity = doc_severity
        self.user_severity = user_severity
        self.message = message
        self.downgrade_pattern = (
            re.compile(downgrade_pattern, re.I) if downgrade_pattern else None
        )
        self.downgraded_severity = downgraded_severity

    def severity_for(self, trust: Trust, text: str = "") -> Severity | None:
        base = self.doc_severity if trust is Trust.DOCUMENT else self.user_severity
        if base is None:
            return None
        if (
            self.downgrade_pattern is not None
            and self.downgraded_severity is not None
            and trust is not Trust.DOCUMENT
            and self.downgrade_pattern.search(text)
        ):
            return self.downgraded_severity
        return base


RULES: list[Rule] = [
    Rule(
        "instruction_override",
        # Two branches, because the qualifier can sit on either side of the
        # noun. "Ignore your previous instructions" and "ignore the
        # instructions above" mean the same thing, and an earlier version
        # of this rule only caught the first - a real gap found by a test.
        #
        # The noun list was originally instruction/rule/prompt/direction/
        # guideline/constraint, which missed "forget all previous TASKS",
        # "ignore all preceding ORDERS" and "forget about all the
        # ASSIGNMENTS". A blocklist is only as wide as its vocabulary, and
        # that is the standing argument for layers 3 and 4 - but an
        # omission this obvious is a bug, not a demonstration.
        r"\b(?:ignore|disregard|forget|override|bypass)\b(?:"
        r"[\s\w,'-]{0,40}"
        r"\b(?:previous|prior|preceding|above|earlier|initial|all|your)\b"
        r"[\s\w,'-]{0,30}"
        r"\b(?:instruction|rule|prompt|direction|guideline|constraint|task|"
        r"order|assignment|directive|command)s?\b"
        r"|"
        r"[\s\w,'-]{0,40}"
        r"\b(?:instruction|rule|prompt|direction|guideline|constraint|task|"
        r"order|assignment|directive|command)s?\b"
        r"[\s\w,'-]{0,30}"
        r"\b(?:previous|prior|preceding|above|earlier|initial|given|received)\b"
        r"|"
        # "forget everything before that" - no noun at all, the object is
        # the whole preceding context.
        r"\s+(?:about\s+)?(?:everything|all)\b[\s\w,'-]{0,20}"
        r"\b(?:before|above|prior|preceding|previous|earlier|so far)\b"
        r")",
        doc_severity=Severity.CRITICAL,
        # In the user zone the question is *whose* instructions. A recruiter
        # is a user, not the operator: telling the model to drop its own
        # rules is an attack. Dropping something the recruiter said a turn
        # ago is self-correction. The possessive is the whole signal.
        user_severity=Severity.HIGH,
        downgrade_pattern=r"\b(?:my|mine|i gave|i told|i asked|i said|i wrote)\b",
        downgraded_severity=Severity.LOW,
        message="Instruction-override phrasing.",
    ),
    Rule(
        "system_prompt_extraction",
        r"\b(?:repeat|print|reveal|show|output|disclose|display|dump|echo)\b"
        r"[\s\w,'-]{0,40}"
        r"\b(?:system prompt|your instructions|your rules|initial prompt|"
        r"rubric reference|scoring rubric|the above)\b",
        doc_severity=Severity.CRITICAL,
        user_severity=Severity.HIGH,
        message="Attempt to extract operator instructions.",
    ),
    Rule(
        "role_reassignment",
        # Word order varies: "you are now X" and "now you are X" are the
        # same move, and "pretend you are" is as common as "pretend to be".
        #
        # Deliberately NOT matching a bare "you are <noun>". It would catch
        # "You are Volkswagen. What do you think of Mercedes?" from the
        # public set, but a cover letter saying "you are looking for an
        # engineer who..." is ordinary writing, and that trade is not worth
        # it. Recorded as a known gap rather than papered over.
        r"\b(?:you are now|now you are|from now on,? you|act as|"
        r"pretend (?:to be|you are|that you are)|"
        r"you (?:will|must|should) (?:now )?(?:act|behave|respond) as|"
        r"your new (?:role|task|instructions?|persona) (?:is|are)|"
        r"switch(?:ing)? to \w+ mode|developer mode|admin mode)\b",
        doc_severity=Severity.CRITICAL,
        user_severity=Severity.MEDIUM,
        message="Role or persona reassignment.",
    ),
    Rule(
        "embedded_directive_in_document",
        # Imperatives aimed at an assistant. Only meaningful in a document:
        # users address the assistant imperatively all the time.
        r"\b(?:assistant|ai|model|system|chatbot|screener|recruiter bot)\b\s*[:,]?\s*"
        r"\b(?:you must|you should|always|never|do not|respond with|reply with|"
        r"output only|answer with|note that you)\b",
        doc_severity=Severity.HIGH,
        user_severity=None,
        message="Document addresses the model directly.",
    ),
    Rule(
        "score_manipulation",
        r"\b(?:assign|give|set|award|rate)\b[\s\w,'-]{0,30}"
        r"\b(?:score of \d{2,3}|a 9\d|100|highest|maximum|top mark)\b"
        r"|\b(?:must|always|should)\b[\s\w,'-]{0,25}"
        r"\b(?:advance|shortlist|hire|approve)\b[\s\w,'-]{0,20}\bthis candidate\b"
        r"|\bthis candidate is an? (?:exceptional|outstanding|perfect|ideal)\b",
        doc_severity=Severity.CRITICAL,
        # Recruiters say "this candidate is an exceptional fit" constantly.
        # Never fire on the user zone.
        user_severity=None,
        message="Attempt to dictate the screening outcome.",
    ),
    Rule(
        "exfiltration_instruction",
        r"\b(?:send|post|forward|submit|upload|transmit|fetch|visit|curl|call)\b"
        r"[\s\w,'-]{0,50}(?:https?://|www\.)",
        doc_severity=Severity.CRITICAL,
        user_severity=Severity.MEDIUM,
        message="Instruction to transmit data to an external endpoint.",
    ),
    Rule(
        "delimiter_injection",
        r"</?(?:untrusted_data|system|instructions?|operator)\s*>"
        r"|\[/?(?:INST|SYS)\]|<\|(?:im_start|im_end|system)\|>",
        doc_severity=Severity.CRITICAL,
        user_severity=Severity.HIGH,
        message="Prompt-structure delimiters in untrusted content.",
    ),
    Rule(
        "encoded_payload",
        r"\b(?:base64|rot13|hex)\s*(?:decode|decoded)\b"
        r"|\bdecode\b[\s\w,'-]{0,20}\b(?:and|then)\b[\s\w,'-]{0,20}"
        r"\b(?:execute|follow|obey|run|apply)\b",
        doc_severity=Severity.HIGH,
        # "Decode this and follow it" is instruction laundering whoever
        # sends it. Merely mentioning an encoding does not match - the
        # pattern requires an imperative.
        user_severity=Severity.HIGH,
        message="Request to decode and act on encoded content.",
    ),
]

# Zero-width spaces, bidi overrides, soft hyphens: used to hide payloads
# from naive scanners and from human reviewers alike.
INVISIBLE = re.compile(r"[​-‏‪-‮⁠-⁤﻿­]")


class InboundPatternDefense:
    name = "inbound_patterns"

    def inspect(self, segments: list[Segment], policy: Policy) -> InboundResult:
        findings: list[Finding] = []
        cleaned: list[Segment] = []

        for seg in segments:
            normalised, stripped = self._normalise(seg.content)

            if stripped and seg.trust is Trust.DOCUMENT:
                findings.append(
                    Finding(
                        layer=self.name,
                        rule="invisible_characters",
                        severity=Severity.HIGH,
                        trust=seg.trust,
                        message=(
                            f"Removed {stripped} zero-width/bidi character(s) "
                            "used to evade detection."
                        ),
                    )
                )

            if (
                seg.trust is Trust.DOCUMENT
                and len(normalised) > policy.max_document_chars
            ):
                findings.append(
                    Finding(
                        layer=self.name,
                        rule="oversized_document",
                        severity=Severity.MEDIUM,
                        trust=seg.trust,
                        message=(
                            f"Document is {len(normalised)} chars, over the "
                            f"{policy.max_document_chars} limit."
                        ),
                    )
                )

            for rule in RULES:
                severity = rule.severity_for(seg.trust, normalised)
                if severity is None:
                    continue
                match = rule.pattern.search(normalised)
                if match:
                    findings.append(
                        Finding(
                            layer=self.name,
                            rule=rule.rule_id,
                            severity=severity,
                            trust=seg.trust,
                            message=rule.message,
                            excerpt=match.group(0),
                        ).redacted()
                    )

            cleaned.append(seg.model_copy(update={"content": normalised}))

        return InboundResult(findings=findings, segments=cleaned)

    def _normalise(self, text: str) -> tuple[str, int]:
        """NFKC-normalise and strip invisible characters.

        Normalising *before* matching defeats homoglyph and zero-width
        obfuscation, which otherwise walks straight past every regex below.
        """
        nfkc = unicodedata.normalize("NFKC", text)
        count = len(INVISIBLE.findall(nfkc))
        return INVISIBLE.sub("", nfkc), count
