"""Turn PDF concealment into gateway findings.

Severity reasoning, stated explicitly because it is a judgement call:

A resume has no legitimate reason to carry text its reader cannot see.
White-on-white fill, sub-point type and off-page positioning are all
deliberate acts by whoever produced the file, so those are CRITICAL.

Invisible render mode is the one exception. Mode 3 is standard in OCR'd
scans, where a text layer is drawn invisibly over the page image so the
document stays searchable. That is entirely normal, and a scanned resume
would trip it. It is still HIGH - which blocks under the default policy -
because in a screening context the asymmetry favours caution, but the
finding says why, so a reviewer can tell the two cases apart.
"""

from __future__ import annotations

from ..models import Finding, Severity, Trust
from .pdf import Concealed, PdfExtraction

SEVERITY: dict[str, Severity] = {
    "white_text": Severity.CRITICAL,
    "micro_font": Severity.CRITICAL,
    "off_page": Severity.CRITICAL,
    "invisible_render_mode": Severity.HIGH,
}

HEADLINE: dict[str, str] = {
    "white_text": "Text painted in white on a white page",
    "micro_font": "Text set far too small to read",
    "off_page": "Text positioned outside the visible page",
    "invisible_render_mode": "Text drawn in an invisible render mode",
}


def to_findings(extraction: PdfExtraction, label: str = "resume.pdf") -> list[Finding]:
    """One finding per distinct concealment technique in the document.

    Grouped by technique rather than per run: a payload split across
    twenty show-text operators is one act of concealment, and twenty
    identical findings would bury everything else in the report.
    """
    by_reason: dict[str, list[Concealed]] = {}
    for c in extraction.concealed:
        by_reason.setdefault(c.reason, []).append(c)

    findings: list[Finding] = []
    for reason, runs in by_reason.items():
        joined = " ".join(r.text for r in runs)
        pages = sorted({r.page for r in runs})
        where = f"page {pages[0]}" if len(pages) == 1 else f"pages {', '.join(map(str, pages))}"
        findings.append(
            Finding(
                layer="ingest",
                rule=f"concealed_text::{reason}",
                severity=SEVERITY.get(reason, Severity.HIGH),
                trust=Trust.DOCUMENT,
                message=(
                    f"{HEADLINE.get(reason, 'Concealed text')} in {label} "
                    f"({where}): {runs[0].detail}. A reader would not have "
                    f"seen this; the model reads it in full."
                ),
                excerpt=joined,
            ).redacted()
        )

    if extraction.truncated:
        findings.append(
            Finding(
                layer="ingest",
                rule="document_truncated",
                severity=Severity.LOW,
                trust=Trust.DOCUMENT,
                message=(
                    f"Only the first {extraction.pages} pages of {label} were "
                    "read; the rest were not screened."
                ),
            )
        )

    return findings
