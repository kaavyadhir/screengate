"""Real Gemini target, on the supported `google-genai` SDK.

Produces the headline numbers against an actual model. Requires
GEMINI_API_KEY. Kept out of CI so the test suite stays offline, free and
deterministic.

A note on model names, because this bit the project once already. The
original code pinned `gemini-2.0-flash`, which Google retired; the first
real run failed with a 404 rather than a benchmark. The default is now
the `gemini-flash-latest` alias, which tracks the current flash model
instead of rotting. The trade is reproducibility: an alias can change
under you, so the benchmark records which model actually answered, and
`SCREENGATE_GEMINI_MODEL` pins an exact version when a run needs to be
repeatable.

Note the prompt assembly: each segment is labelled with its provenance.
That labelling is the baseline the spotlighting layer improves on - even
undefended, a well-built app says which part is the resume. The attack
still works; that is the point.
"""

from __future__ import annotations

import os
import random
import re
import time
from typing import Any, Callable, TypeVar

from ..models import Segment, Trust

T = TypeVar("T")

_HEADER = {
    Trust.USER: "Recruiter request",
    Trust.DOCUMENT: "Candidate resume",
    Trust.OPERATOR: "Job description",
    Trust.MODEL: "Prior model output",
}

# Free-tier Gemini is rate limited to a handful of requests per minute, and
# a benchmark run is hundreds of calls. Without backoff the run dies
# halfway through and the partial numbers are worse than none.
RATE_LIMIT_MARKERS = ("429", "rate limit", "resource_exhausted", "quota", "unavailable")

# Google returns 429 for two completely different conditions:
#
#   per-minute throttling    - transient, clears in seconds, retrying works
#   per-DAY quota exhausted  - lasts until a reset hours away, retrying is futile
#
# A substring match on "429" or "quota" cannot tell them apart, which is
# how a deployed request spent ~30s on exponential backoff before failing
# anyway - on an error whose own payload said "Please retry in 14h18m44s".
# The response carries the answer, so read it instead of guessing.
RETRY_DELAY_RE = re.compile(
    r"retrydelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s"   # retryDelay: '51524s'
    r"|please retry in\s+(?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?",
    re.I,
)

# Longest server-requested delay still worth waiting out in-process.
MAX_WAIT_S = 120.0


def _server_requested_delay(msg: str) -> float | None:
    """How many seconds the API asked us to wait, if it said at all."""
    m = RETRY_DELAY_RE.search(msg)
    if not m:
        return None
    if m.group(1):
        return float(m.group(1))
    hours, mins, secs = m.group(2), m.group(3), m.group(4)
    if not any((hours, mins, secs)):
        return None
    return float(hours or 0) * 3600 + float(mins or 0) * 60 + float(secs or 0)


class QuotaExhausted(RuntimeError):
    """A rate limit that will not clear in any useful timeframe.

    Raised instead of retrying, so callers fail fast and for a stated
    reason. The semantic layer turns this into a degraded-layer finding
    rather than a 500; a benchmark stops rather than limping.
    """


def with_retries(fn: Callable[[], T], attempts: int = 5) -> T:
    last: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - provider SDK raises broadly
            msg = str(exc).lower()
            if not any(m in msg for m in RATE_LIMIT_MARKERS):
                raise

            wanted = _server_requested_delay(str(exc))
            if wanted is not None and wanted > MAX_WAIT_S:
                raise QuotaExhausted(
                    f"Quota exhausted: the API asks for {wanted / 3600:.1f}h "
                    "before retrying, so this is not transient throttling and "
                    "retrying would only waste time. Wait for the reset, or "
                    "switch SCREENGATE_PROVIDER / SCREENGATE_EMBEDDER."
                ) from exc

            if i == attempts - 1:
                raise
            last = exc
            # Honour a short server-stated delay over our own guess; the
            # server knows when its window actually reopens.
            delay = min(2**i, 30) + random.uniform(0, 1)
            if wanted is not None:
                delay = max(delay, min(wanted, MAX_WAIT_S))
            print(f"  rate limited, retrying in {delay:.1f}s ...", flush=True)
            time.sleep(delay)
    raise last if last else RuntimeError("retry loop exhausted")


def build_client(api_key: str | None = None):
    """Create a google-genai client, with a readable error if it cannot."""
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Put it in .env, or run with "
            "SCREENGATE_PROVIDER=mock to use the offline target."
        )
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "google-genai is not installed. pip install -r requirements.txt\n"
            "(The older google-generativeai package is end-of-life and is "
            "no longer used here.)"
        ) from exc
    return genai.Client(api_key=key)


def _generation_config(temperature: float, json_out: bool) -> Any:
    """Build a config object, tolerating SDK shape changes.

    The typed config classes move around between SDK versions. A plain
    dict is accepted by every version seen so far, so it is the fallback
    rather than the primary - and if both fail the caller runs without a
    config rather than crashing the benchmark.
    """
    fields: dict[str, Any] = {"temperature": temperature}
    if json_out:
        fields["response_mime_type"] = "application/json"
    try:
        from google.genai import types

        return types.GenerateContentConfig(**fields)
    except Exception:
        return fields


class GeminiProvider:
    name = "gemini"

    def __init__(
        self, api_key: str | None = None, model: str = "gemini-flash-latest"
    ) -> None:
        self._client = build_client(api_key)
        self.model_name = model
        self.resolved_model: str | None = None
        """Which model actually answered - recorded so an alias stays traceable."""

    def complete(self, system: str, segments: list[Segment]) -> str:
        parts = [system]
        for seg in segments:
            header = _HEADER.get(seg.trust, "Input")
            if seg.label:
                header = f"{header} ({seg.label})"
            parts.append(f"{header}:\n{seg.content}")
        prompt = "\n\n".join(parts)

        def call():
            return self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=_generation_config(0.0, json_out=True),
            )

        try:
            resp = with_retries(call)
        except Exception as exc:
            msg = str(exc).lower()
            if "404" in msg or "not found" in msg or "no longer available" in msg:
                raise RuntimeError(
                    f"Model {self.model_name!r} is not available to this key — "
                    "most likely retired.\n"
                    "Find one that works:  python -m tools.check_models\n"
                    "then set SCREENGATE_GEMINI_MODEL in .env."
                ) from exc
            if "503" in msg or "unavailable" in msg or "high demand" in msg:
                raise RuntimeError(
                    f"Model {self.model_name!r} is overloaded on Google's side "
                    "(503), not misconfigured.\n"
                    "This is capacity, not your key or this code. Either wait, "
                    "or pick a less busy model:\n"
                    "  python -m tools.check_models\n"
                    "A specific older version is often freer than whatever the "
                    "'-latest' alias points at."
                ) from exc
            raise

        if self.resolved_model is None:
            self.resolved_model = (
                getattr(resp, "model_version", None) or self.model_name
            )

        # A safety-filtered or empty response has no usable text. That is
        # not a crash and not a pass: it is output that fails the
        # contract, which is exactly what layer 3 should see.
        try:
            return (resp.text or "").strip()
        except Exception:
            return ""
