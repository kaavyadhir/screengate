#!/usr/bin/env python3
"""Find a Gemini model that is actually answering right now.

This project has been bitten twice by model availability: once by a
retired model returning 404, once by a live model returning 503 under
load. Both failed at the start of a benchmark run rather than before it,
which is the expensive moment to find out.

So: probe candidates with one cheap call each, report which respond and
how fast, and print the line to paste into .env.

    python -m tools.check_models
    python -m tools.check_models --embeddings
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout

from screengate.config import Settings

# The SDK retries internally with its own backoff, so a probe of an
# overloaded model can stall for a long time before raising. A probe is
# meant to be fast and answer "is this usable right now", so each call
# gets a hard ceiling and the loop carries on regardless.
PROBE_TIMEOUT_S = 20.0


def say(*parts: object) -> None:
    """print with flush: Windows buffers, and a probe that appears to
    hang is indistinguishable from one that has."""
    print(*parts, flush=True)

# Ordered by preference. Aliases first - they track the current model and
# do not rot - then pinned versions as fallbacks, since a specific older
# model is often less loaded than whatever the alias points at.
CHAT_CANDIDATES = (
    "gemini-flash-latest",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
)

EMBED_CANDIDATES = (
    "gemini-embedding-001",
    "gemini-embedding-2",
    "gemini-embedding-2-preview",
)


def classify(exc: Exception) -> str:
    msg = str(exc).lower()
    if "404" in msg or "not found" in msg or "no longer available" in msg:
        return "retired"
    if "503" in msg or "unavailable" in msg or "high demand" in msg:
        return "overloaded"
    if "429" in msg or "quota" in msg or "resource_exhausted" in msg:
        return "rate limited"
    if "permission" in msg or "403" in msg or "api key" in msg:
        return "not permitted"
    return type(exc).__name__


def probe_chat(client, model: str) -> tuple[bool, str, float]:
    started = time.perf_counter()
    try:
        resp = client.models.generate_content(
            model=model, contents="Reply with exactly: OK"
        )
        dt = time.perf_counter() - started
        text = (getattr(resp, "text", "") or "").strip()
        return True, text[:20] or "(empty)", dt
    except Exception as exc:  # noqa: BLE001
        return False, classify(exc), time.perf_counter() - started


def probe_embed(client, model: str) -> tuple[bool, str, float]:
    started = time.perf_counter()
    try:
        resp = client.models.embed_content(model=model, contents=["hello"])
        dt = time.perf_counter() - started
        items = getattr(resp, "embeddings", None) or []
        dims = len(getattr(items[0], "values", items[0])) if items else 0
        return True, f"{dims} dims", dt
    except Exception as exc:  # noqa: BLE001
        return False, classify(exc), time.perf_counter() - started


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe Gemini model availability")
    ap.add_argument("--embeddings", action="store_true", help="probe embedding models")
    args = ap.parse_args()

    from screengate.providers.gemini import build_client

    settings = Settings.from_env()
    client = build_client(settings.gemini_api_key)

    candidates = EMBED_CANDIDATES if args.embeddings else CHAT_CANDIDATES
    probe = probe_embed if args.embeddings else probe_chat
    env_key = (
        "SCREENGATE_EMBEDDING_MODEL" if args.embeddings else "SCREENGATE_GEMINI_MODEL"
    )

    kind = "embedding" if args.embeddings else "chat"
    say(f"probing {len(candidates)} {kind} models\n")
    working: list[tuple[str, float]] = []
    reasons: dict[str, int] = {}

    for model in candidates:
        started = time.perf_counter()
        # A fresh executor per probe, not one shared pool: a model that
        # hangs leaves its worker stuck, and anything queued behind it
        # would report "timed out" too - which would misdiagnose every
        # later model. The stranded thread is a daemon and dies with the
        # process, which is fine for a short-lived probe.
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(probe, client, model)
            try:
                ok, detail, dt = future.result(timeout=PROBE_TIMEOUT_S)
            except FuturesTimeout:
                ok, detail, dt = False, "timed out", time.perf_counter() - started
            except Exception as exc:  # noqa: BLE001 - never stop the loop
                ok, detail, dt = False, classify(exc), time.perf_counter() - started
        finally:
            pool.shutdown(wait=False)

        say(f"  {'OK    ' if ok else 'FAIL  '} {model:28} {dt:5.2f}s  {detail}")
        if ok:
            working.append((model, dt))
        else:
            reasons[detail] = reasons.get(detail, 0) + 1

    say()
    if not working:
        summary = ", ".join(f"{n}x {r}" for r, n in sorted(reasons.items()))
        say(f"Nothing responded ({summary}).")
        if reasons.get("overloaded") or reasons.get("timed out"):
            say("Overloaded is Google's capacity, not your setup or this code.")
            say("Wait and re-run; demand spikes are usually short.")
        if reasons.get("not permitted"):
            say("'not permitted' points at the key in .env.")
        if reasons.get("retired"):
            say("'retired' means the name is gone; list what exists with:")
            say("  python -c \"from google import genai; from screengate.config "
                "import Settings; c=genai.Client(api_key=Settings.from_env()"
                ".gemini_api_key); print([m.name for m in c.models.list()])\"")
        return 1

    best = min(working, key=lambda w: w[1])[0]
    say(f"{len(working)} of {len(candidates)} responding. Fastest: {best}")
    say(f"\nPut this in .env:\n  {env_key}={best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
