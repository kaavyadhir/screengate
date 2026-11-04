"""Tests for PDF ingestion and concealment detection.

The sample PDFs are generated rather than committed as fixtures, so the
tests exercise the same files the demo serves and a change to the
generator cannot silently drift from what is being tested.
"""

from __future__ import annotations

from screengate.config import LayerToggles, Policy, Settings
from screengate.ingest import PdfTooLarge, PdfUnreadable, extract, to_findings
from screengate.ingest.pdf import scan_stream
from screengate.models import Decision, GatewayRequest, Segment, Severity, Trust
from screengate.pipeline import Gateway
from screengate.providers.mock import MockProvider
from screengate.rag.embeddings import HashingEmbedder
from tools.make_sample_pdfs import (
    PAYLOAD,
    RESUME,
    build_pdf,
    invisible_mode,
    tiny_offpage,
    visible_block,
    white_text,
)

BODY = visible_block([l for l in RESUME if l[0]])
A4 = (0.0, 0.0, 595.0, 842.0)


def _pdf(extra: str = "") -> bytes:
    return build_pdf(BODY + ("\n" + extra if extra else ""))


def _reasons(data: bytes) -> set[str]:
    return {c.reason for c in extract(data).concealed}


# ---- extraction --------------------------------------------------------


def test_clean_pdf_extracts_text_and_hides_nothing():
    res = extract(_pdf())
    assert res.pages == 1
    assert "Arjun Mehta" in res.text
    assert res.concealed == []
    assert res.scan_error is None


def test_payload_is_extracted_even_though_invisible():
    """The whole problem in one assertion.

    A human opening white_text.pdf sees a normal resume. The parser - and
    therefore the model - reads the instruction in full.
    """
    res = extract(_pdf(white_text(PAYLOAD, 60, 300)))
    assert "Ignore previous instructions" in res.text


def test_rejects_non_pdf():
    try:
        extract(b"this is not a pdf" * 10)
    except PdfUnreadable:
        return
    raise AssertionError("a non-PDF should be rejected")


def test_malformed_inputs_reject_rather_than_crash():
    """Everything a stranger might upload lands on a clean error path."""
    hostile = {
        "png": b"\x89PNG\r\n\x1a\n" + b"\x00" * 2000,
        "zip": b"PK\x03\x04" + b"\x00" * 2000,
        "empty": b"",
        "header only": b"%PDF-1.4\n",
        "truncated": _pdf()[:400],
    }
    for name, data in hostile.items():
        try:
            extract(data)
        except (PdfUnreadable, PdfTooLarge):
            continue
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"{name} crashed with {type(exc).__name__}: {exc}")
        raise AssertionError(f"{name} should not have parsed")


def test_tokenizer_survives_hostile_streams():
    """Malformed operators must not take the scan down with them."""
    for stream in (
        b"BT /F1 9 Tf (" + b"(" * 200 + b")" * 200 + b") Tj ET",  # nested parens
        b"BT (unterminated",                                      # no closing
        b"Q Q Q Q BT /F1 9 Tf (x) Tj ET",                         # stack underflow
        b"1 1 1 rg BT <48656c6c6f> Tj ET",                        # hex string
        b"/F1 9 Tf ... ??? \x00\xff garbage",                     # junk
    ):
        scan_stream(stream, 1, A4)


def test_rejects_oversized_file():
    try:
        extract(b"%PDF-1.4\n" + b"x" * 5000, max_bytes=1000)
    except PdfTooLarge:
        return
    raise AssertionError("an oversized file should be rejected")


def test_page_cap_sets_truncated():
    res = extract(_pdf(), max_pages=0)
    assert res.truncated is True


# ---- concealment detection --------------------------------------------


def test_detects_white_text():
    assert "white_text" in _reasons(_pdf(white_text(PAYLOAD, 60, 300)))


def test_detects_invisible_render_mode():
    assert "invisible_render_mode" in _reasons(_pdf(invisible_mode(PAYLOAD, 60, 300)))


def test_detects_micro_font():
    assert "micro_font" in _reasons(_pdf(tiny_offpage(PAYLOAD)))


def test_detects_off_page_at_readable_size():
    """Off-page positioning on its own, without the tiny-font signal."""
    stream = (
        "BT /F1 10 Tf 1 0 0 1 2400 -600 Tm (hidden over here) Tj ET"
    ).encode()
    reasons = {c.reason for c in scan_stream(stream, 1, A4)}
    assert "off_page" in reasons


def test_gray_white_is_detected():
    """`1 g` is white in DeviceGray - the same trick, a different operator."""
    stream = b"1 g BT /F1 9 Tf 1 0 0 1 60 300 Tm (hidden) Tj ET"
    assert "white_text" in {c.reason for c in scan_stream(stream, 1, A4)}


def test_cmyk_white_is_detected():
    stream = b"0 0 0 0 k BT /F1 9 Tf 1 0 0 1 60 300 Tm (hidden) Tj ET"
    assert "white_text" in {c.reason for c in scan_stream(stream, 1, A4)}


def test_graphics_state_is_restored_by_Q():
    """White set inside q/Q must not leak onto text drawn after it."""
    stream = (
        b"q 1 1 1 rg Q "
        b"BT /F1 10 Tf 1 0 0 1 60 300 Tm (perfectly normal text) Tj ET"
    )
    assert scan_stream(stream, 1, A4) == []


def test_black_text_is_not_flagged():
    stream = b"0 0 0 rg BT /F1 11 Tf 1 0 0 1 60 300 Tm (ordinary resume line) Tj ET"
    assert scan_stream(stream, 1, A4) == []


def test_tiny_scale_via_text_matrix_is_caught():
    """Font size 12 scaled to near-nothing by the text matrix."""
    stream = b"BT /F1 12 Tf 0.05 0 0 0.05 60 300 Tm (scaled down to nothing) Tj ET"
    assert "micro_font" in {c.reason for c in scan_stream(stream, 1, A4)}


def test_decompression_is_bounded():
    """A deliberately explosive content stream must not be inflated whole.

    PDF streams are Flate-compressed and ratios of several hundred to one
    are trivial to produce: 104 KB expanding to 42 MB was measured. Inside
    a 5 MB upload cap that is gigabytes of memory from one request.
    """
    import zlib

    from screengate.ingest.pdf import MAX_STREAM_BYTES, _bounded_inflate

    payload = b"BT /F1 9 Tf (x) Tj ET\n" * 2_000_000
    out = _bounded_inflate(zlib.compress(payload, 9))
    assert len(out) <= MAX_STREAM_BYTES
    assert len(out) < len(payload)


def test_uncompressed_stream_passes_through():
    from screengate.ingest.pdf import _bounded_inflate

    raw = b"BT /F1 9 Tf 1 0 0 1 60 300 Tm (visible) Tj ET"
    assert _bounded_inflate(raw) == raw


def test_run_flood_is_capped():
    """Tens of thousands of hidden runs must not become a report bomb."""
    from screengate.ingest.pdf import MAX_RUNS_PER_PAGE

    stream = b"1 1 1 rg " + b"BT /F1 9 Tf 1 0 0 1 60 300 Tm (h) Tj ET " * 5000
    assert len(scan_stream(stream, 1, A4)) == MAX_RUNS_PER_PAGE


def test_whitespace_only_runs_are_ignored():
    stream = b"1 1 1 rg BT /F1 9 Tf 1 0 0 1 60 300 Tm (   ) Tj ET"
    assert scan_stream(stream, 1, A4) == []


def test_tj_array_text_is_assembled():
    stream = b"1 1 1 rg BT /F1 9 Tf 1 0 0 1 60 300 Tm [(Ig)-20(nore this)] TJ ET"
    found = scan_stream(stream, 1, A4)
    assert found and "nore this" in found[0].text


# ---- findings ----------------------------------------------------------


def test_findings_are_critical_and_grouped():
    findings = to_findings(extract(_pdf(white_text(PAYLOAD, 60, 300))))
    assert len(findings) == 1
    assert findings[0].rule == "concealed_text::white_text"
    assert findings[0].severity is Severity.CRITICAL
    assert findings[0].trust is Trust.DOCUMENT


def test_invisible_mode_is_high_not_critical():
    """OCR layers use mode 3 legitimately; the severity reflects that."""
    findings = to_findings(extract(_pdf(invisible_mode(PAYLOAD, 60, 300))))
    assert findings[0].severity is Severity.HIGH


def test_clean_pdf_produces_no_findings():
    assert to_findings(extract(_pdf())) == []


def test_many_runs_collapse_into_one_finding():
    extra = "\n".join(white_text(f"part {i}", 60, 300 - i * 10) for i in range(8))
    findings = to_findings(extract(_pdf(extra)))
    assert len(findings) == 1


# ---- end to end --------------------------------------------------------


def _gateway() -> Gateway:
    return Gateway(
        provider=MockProvider(),
        policy=Policy(),
        settings=Settings(provider="mock", embedder="hashing"),
        embedder=HashingEmbedder(),
    )


def _screen(data: bytes, layers: LayerToggles) -> Decision:
    res = extract(data)
    req = GatewayRequest(
        user_input="Screen this candidate against the role.",
        documents=[Segment(trust=Trust.DOCUMENT, content=res.text, label="r.pdf")],
    )
    return _gateway().handle(
        req, policy=Policy().with_layers(layers), prior_findings=to_findings(res)
    ).decision


def test_clean_pdf_passes_the_gateway():
    assert _screen(_pdf(), LayerToggles()) is Decision.ALLOW


def test_concealed_pdf_is_blocked():
    assert _screen(_pdf(white_text(PAYLOAD, 60, 300)), LayerToggles()) is Decision.BLOCK


def test_concealment_blocks_with_every_layer_disabled():
    """Ingest detection is independent of the four defense layers.

    This is the property worth protecting: the file is rejected for how it
    was built, before any layer forms an opinion about what it says.
    """
    assert (
        _screen(_pdf(white_text(PAYLOAD, 60, 300)), LayerToggles.none())
        is Decision.BLOCK
    )


def test_blocked_upload_never_reaches_the_model():
    calls: list[int] = []

    class Counting(MockProvider):
        def complete(self, system, segments):
            calls.append(1)
            return super().complete(system, segments)

    res = extract(_pdf(white_text(PAYLOAD, 60, 300)))
    gw = Gateway(
        provider=Counting(),
        policy=Policy(),
        settings=Settings(provider="mock", embedder="hashing"),
        embedder=HashingEmbedder(),
    )
    gw.handle(
        GatewayRequest(
            user_input="Screen this.",
            documents=[Segment(trust=Trust.DOCUMENT, content=res.text)],
        ),
        prior_findings=to_findings(res),
    )
    assert calls == [], "model was called on a file that was already blocking"
