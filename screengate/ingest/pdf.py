"""PDF ingestion, with concealment detection.

Text extraction alone is not enough here. The interesting property of a
weaponised resume is not that it *contains* an instruction - it is that a
human reviewer cannot see the instruction while the model reads it
perfectly. Any difference between what a person sees and what a parser
extracts is, by itself, a strong signal, independent of what the hidden
text actually says.

So this module does two things:

  1. Extracts text (pypdf), which is what the gateway then screens.
  2. Scans the raw content stream for text that has been deliberately
     made invisible, and reports each occurrence.

The second is a small interpreter over the PDF content stream, tracking
enough graphics state to answer "would a reader have seen this?":

  invisible_render_mode  text render mode 3 or 7 - draw nothing
  white_text             fill colour is white (or near-white) on a white page
  micro_font             effective size under 1.5pt - present but unreadable
  off_page               positioned outside the page box

HONEST LIMITS. This is a heuristic scan, not a rendering engine. It will
miss text hidden by other means: covered by an opaque image or filled
rectangle, clipped away by a path, placed behind another layer, or set in
a colour that merely matches a coloured background rather than being
white. A negative result means "none of the four checked techniques",
never "nothing is hidden". It can also produce false positives: OCR
layers over scanned documents legitimately use render mode 3, which is
why that finding is reported with its reason attached rather than as a
bare verdict.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field, replace
from typing import Iterator

MAX_BYTES = 5 * 1024 * 1024
MAX_PAGES = 15

# Ceiling on a single decompressed content stream. PDF streams are
# Flate-compressed, and compression ratios of several hundred to one are
# easy to produce deliberately: a 100 KB stream expanding to 40 MB was
# measured while writing this. Without a bound, one upload inside the
# size cap can exhaust memory, so the stream is decompressed
# incrementally and abandoned at the limit.
MAX_STREAM_BYTES = 8 * 1024 * 1024

# Ceiling on reported concealed runs per page. A document emitting tens of
# thousands of one-character hidden runs is itself a denial-of-service on
# whatever renders the report.
MAX_RUNS_PER_PAGE = 200

# Render modes that paint nothing. 3 is "neither fill nor stroke";
# 7 is "add to clipping path only".
INVISIBLE_MODES = {3, 7}

# Below this effective point size, text is present but not readable.
MICRO_FONT_PT = 1.5

# How close to pure white counts as white, per channel (0..1).
WHITE_EPSILON = 0.06


@dataclass
class Concealed:
    """One run of text that a reader would not have seen."""

    text: str
    reason: str
    detail: str
    page: int

    def short(self, limit: int = 160) -> str:
        t = " ".join(self.text.split())
        return t if len(t) <= limit else t[:limit] + "..."


@dataclass
class PdfExtraction:
    text: str
    pages: int
    concealed: list[Concealed] = field(default_factory=list)
    truncated: bool = False
    scan_error: str | None = None
    """Set when the concealment scan failed; extraction may still have worked."""

    @property
    def has_concealed(self) -> bool:
        return bool(self.concealed)


class PdfTooLarge(ValueError):
    pass


class PdfUnreadable(ValueError):
    pass


# ---------------------------------------------------------------- tokens

TOKEN_RE = re.compile(
    rb"""
      (?P<name>/[^\s/\[\]<>(){}%]*)
    | (?P<num>[-+]?\d*\.?\d+)
    | (?P<hex><[0-9A-Fa-f\s]*>)
    | (?P<open>\[) | (?P<close>\])
    | (?P<dict_open><<) | (?P<dict_close>>>)
    | (?P<op>[A-Za-z'"*][A-Za-z0-9'"*]*)
    """,
    re.X,
)


def _tokenize(data: bytes) -> Iterator[tuple[str, object]]:
    """Yield (kind, value) over a content stream.

    Literal strings are handled separately from the regex because they
    nest and escape, which a regex cannot track reliably.
    """
    i, n = 0, len(data)
    while i < n:
        ch = data[i : i + 1]

        if ch in b" \t\r\n\f\x00":
            i += 1
            continue

        if ch == b"%":  # comment to end of line
            j = data.find(b"\n", i)
            i = n if j < 0 else j + 1
            continue

        if ch == b"(":
            text, i = _read_literal(data, i)
            yield ("str", text)
            continue

        m = TOKEN_RE.match(data, i)
        if not m:
            i += 1
            continue
        i = m.end()

        if m.lastgroup == "num":
            try:
                yield ("num", float(m.group()))
            except ValueError:
                pass
        elif m.lastgroup == "hex":
            yield ("str", _read_hex(m.group()))
        elif m.lastgroup == "name":
            yield ("name", m.group().decode("latin-1", "replace"))
        elif m.lastgroup in ("open", "close", "dict_open", "dict_close"):
            yield (m.lastgroup, None)
        else:
            yield ("op", m.group().decode("latin-1", "replace"))


def _read_literal(data: bytes, i: int) -> tuple[str, int]:
    """Read a (...) string starting at i, returning the text and new index."""
    i += 1
    depth = 1
    out = bytearray()
    n = len(data)
    while i < n:
        c = data[i : i + 1]
        if c == b"\\":
            nxt = data[i + 1 : i + 2]
            mapping = {
                b"n": b"\n", b"r": b"\r", b"t": b"\t",
                b"b": b"\b", b"f": b"\f",
                b"(": b"(", b")": b")", b"\\": b"\\",
            }
            if nxt in mapping:
                out += mapping[nxt]
                i += 2
                continue
            if nxt.isdigit():  # octal escape
                j = i + 1
                digits = b""
                while j < n and len(digits) < 3 and data[j : j + 1].isdigit():
                    digits += data[j : j + 1]
                    j += 1
                try:
                    out.append(int(digits, 8) & 0xFF)
                except ValueError:
                    pass
                i = j
                continue
            i += 2
            continue
        if c == b"(":
            depth += 1
            out += c
        elif c == b")":
            depth -= 1
            if depth == 0:
                return out.decode("latin-1", "replace"), i + 1
            out += c
        else:
            out += c
        i += 1
    return out.decode("latin-1", "replace"), i


def _read_hex(token: bytes) -> str:
    digits = re.sub(rb"[^0-9A-Fa-f]", b"", token)
    if len(digits) % 2:
        digits += b"0"
    try:
        return bytes.fromhex(digits.decode("ascii")).decode("latin-1", "replace")
    except ValueError:
        return ""


# ------------------------------------------------------------ the scan


@dataclass(frozen=True)
class _State:
    fill: tuple[float, float, float] = (0.0, 0.0, 0.0)
    render_mode: int = 0
    font_size: float = 12.0
    scale: float = 1.0
    x: float = 0.0
    y: float = 0.0


def _is_white(rgb: tuple[float, float, float]) -> bool:
    return all(c >= 1.0 - WHITE_EPSILON for c in rgb)


def scan_stream(
    data: bytes, page_no: int, page_box: tuple[float, float, float, float]
) -> list[Concealed]:
    """Walk one page's content stream and report concealed text runs."""
    st = _State()
    stack: list[_State] = []
    pending: list[float] = []
    found: list[Concealed] = []
    x0, y0, x1, y1 = page_box

    def numbers(count: int) -> list[float]:
        vals = [v for v in pending[-count:] if isinstance(v, float)]
        return vals if len(vals) == count else []

    def judge(text: str) -> None:
        if not text.strip() or len(found) >= MAX_RUNS_PER_PAGE:
            return
        effective = abs(st.font_size * st.scale)

        if st.render_mode in INVISIBLE_MODES:
            found.append(
                Concealed(
                    text, "invisible_render_mode",
                    f"text render mode {st.render_mode} paints nothing",
                    page_no,
                )
            )
            return
        if _is_white(st.fill):
            r, g, b = st.fill
            found.append(
                Concealed(
                    text, "white_text",
                    f"fill colour rgb({r:.2f}, {g:.2f}, {b:.2f}) on a white page",
                    page_no,
                )
            )
            return
        if effective < MICRO_FONT_PT:
            found.append(
                Concealed(
                    text, "micro_font",
                    f"effective size {effective:.2f}pt is below {MICRO_FONT_PT}pt",
                    page_no,
                )
            )
            return
        if not (x0 - 1 <= st.x <= x1 + 1 and y0 - 1 <= st.y <= y1 + 1):
            found.append(
                Concealed(
                    text, "off_page",
                    f"drawn at ({st.x:.0f}, {st.y:.0f}), outside the page box",
                    page_no,
                )
            )

    for kind, value in _tokenize(data):
        if kind in ("num", "str", "name"):
            pending.append(value)
            continue
        if kind != "op":
            continue

        op = value

        if op == "q":
            stack.append(st)
        elif op == "Q":
            st = stack.pop() if stack else _State()
        elif op == "rg":
            v = numbers(3)
            if v:
                st = replace(st, fill=(v[0], v[1], v[2]))
        elif op == "g":
            v = numbers(1)
            if v:
                st = replace(st, fill=(v[0], v[0], v[0]))
        elif op == "k":
            v = numbers(4)
            if v:
                c, m, y_, k = v
                st = replace(
                    st,
                    fill=(
                        (1 - min(1.0, c + k)),
                        (1 - min(1.0, m + k)),
                        (1 - min(1.0, y_ + k)),
                    ),
                )
        elif op == "Tf":
            v = numbers(1)
            if v:
                st = replace(st, font_size=v[0])
        elif op == "Tr":
            v = numbers(1)
            if v:
                st = replace(st, render_mode=int(v[0]))
        elif op == "Tm":
            v = numbers(6)
            if v:
                a, b, c, d, e, f = v
                st = replace(st, scale=max(abs(a), abs(d)) or 1.0, x=e, y=f)
        elif op in ("Td", "TD"):
            v = numbers(2)
            if v:
                st = replace(st, x=st.x + v[0], y=st.y + v[1])
        elif op == "BT":
            st = replace(st, x=0.0, y=0.0, scale=1.0)
        elif op in ("Tj", "'", '"'):
            texts = [p for p in pending if isinstance(p, str) and not p.startswith("/")]
            if texts:
                judge(texts[-1])
        elif op == "TJ":
            parts = [p for p in pending if isinstance(p, str) and not p.startswith("/")]
            if parts:
                judge("".join(parts))

        pending.clear()

    return found


# ------------------------------------------------------------- extract


def extract(
    data: bytes, *, max_bytes: int = MAX_BYTES, max_pages: int = MAX_PAGES
) -> PdfExtraction:
    """Extract text from a PDF and report any concealed runs.

    Caps are enforced before parsing: this endpoint accepts files from
    strangers, and an unbounded parser is its own denial-of-service.
    """
    if len(data) > max_bytes:
        raise PdfTooLarge(
            f"File is {len(data) // 1024} KB; the limit is {max_bytes // 1024} KB."
        )
    if not data[:1024].lstrip().startswith(b"%PDF"):
        raise PdfUnreadable("That does not look like a PDF file.")

    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise PdfUnreadable(
            "pypdf is not installed. pip install -r requirements.txt"
        ) from exc

    import io

    try:
        reader = PdfReader(io.BytesIO(data))
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception as exc:
                raise PdfUnreadable("The PDF is password protected.") from exc
        pages = list(reader.pages)
    except PdfUnreadable:
        raise
    except Exception as exc:
        raise PdfUnreadable(f"Could not read the PDF: {exc}") from exc

    truncated = len(pages) > max_pages
    pages = pages[:max_pages]

    chunks: list[str] = []
    concealed: list[Concealed] = []
    scan_error: str | None = None

    for n, page in enumerate(pages, start=1):
        try:
            chunks.append(page.extract_text() or "")
        except Exception:
            chunks.append("")

        # The concealment scan is best-effort: a malformed stream must not
        # cost us the extracted text, which is the part the gateway needs.
        try:
            box = _page_box(page)
            raw = page.get_contents()
            if raw is not None:
                blob = _bounded_inflate(raw.get_data())
                concealed.extend(scan_stream(blob, n, box))
        except Exception as exc:
            scan_error = str(exc)

    return PdfExtraction(
        text="\n".join(chunks).strip(),
        pages=len(pages),
        concealed=concealed,
        truncated=truncated,
        scan_error=scan_error,
    )


def _bounded_inflate(blob: bytes, limit: int = MAX_STREAM_BYTES) -> bytes:
    """Decompress a content stream, abandoning it past `limit` bytes.

    pypdf normally hands back already-decoded data, but a stream can
    arrive still deflated. Decompressing incrementally and stopping at the
    ceiling means a deliberately explosive stream costs `limit` bytes
    rather than whatever the attacker chose. Truncation only shortens the
    concealment scan; extraction is pypdf's job and is unaffected.
    """
    if not blob[:1] == b"\x78":
        return blob
    try:
        d = zlib.decompressobj()
        out = d.decompress(blob, limit)
        return out if out else blob
    except zlib.error:
        return blob


def _page_box(page) -> tuple[float, float, float, float]:
    try:
        box = page.mediabox
        return (float(box.left), float(box.bottom), float(box.right), float(box.top))
    except Exception:
        return (0.0, 0.0, 595.0, 842.0)
