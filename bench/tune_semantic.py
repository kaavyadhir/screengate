#!/usr/bin/env python3
"""Pick the semantic threshold from data instead of guessing it.

The 0.72 default was a guess. A first real measurement put a paraphrase
attack at 0.772 and an ordinary resume line at 0.537 - so the threshold
sits between them, but with only 0.05 of headroom above the attack. A
slightly different wording scoring 0.70 would walk straight through.

This embeds every case in the corpus, computes each one's peak
similarity to the canonical attack intents, and sweeps thresholds to
show the trade. Output is the separation between the two populations and
the threshold that maximises it.

METHODOLOGY. This tunes on the LOCAL corpus, which is self-authored.
That is legitimate - it is a design parameter, not a result - but it
means the number this prints is not evidence the layer works. The
held-out set is what tests that, and it must be re-run after any change
here:

    python -m bench.heldout.run_heldout --offset 200

Needs embeddings but not chat, so it runs when the chat models are
overloaded.

    python -m bench.tune_semantic
    python -m bench.tune_semantic --json bench/semantic_tuning.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from screengate.config import Settings
from screengate.defenses.semantic import SemanticDefense
from screengate.models import Trust
from screengate.rag.embeddings import build_embedder

from .runner import load_corpus

SWEEP = [round(0.50 + 0.01 * i, 2) for i in range(41)]  # 0.50 .. 0.90


@dataclass
class Scored:
    case_id: str
    kind: str
    peak: float
    intent: str
    category: str | None


def score_corpus(defense: SemanticDefense) -> list[Scored]:
    out: list[Scored] = []
    cases = load_corpus()
    for n, case in enumerate(cases, start=1):
        # Document-zone text only. Recruiter instructions are supposed to
        # look like instructions; scoring them against attack intents
        # would flag every normal request.
        text = "\n".join(
            d.content for d in case.documents if d.trust is Trust.DOCUMENT
        )
        if not text.strip():
            text = case.user_input
        peak, intent = defense.peak_similarity(text)
        out.append(
            Scored(case.id, case.kind, round(peak, 4), intent, case.category or case.tripwire)
        )
        print(f"  [{n:>2}/{len(cases)}] {case.kind:7} {case.id:38} {peak:.3f}  {intent}",
              flush=True)
    return out


def sweep(scored: list[Scored]) -> list[dict]:
    attacks = [s for s in scored if s.kind == "attack"]
    benign = [s for s in scored if s.kind == "benign"]
    rows = []
    for t in SWEEP:
        tp = sum(1 for s in attacks if s.peak >= t)
        fp = sum(1 for s in benign if s.peak >= t)
        rows.append(
            {
                "threshold": t,
                "attacks_caught": tp,
                "attacks_total": len(attacks),
                "recall_pct": round(100 * tp / len(attacks), 1) if attacks else 0.0,
                "benign_blocked": fp,
                "benign_total": len(benign),
                "fpr_pct": round(100 * fp / len(benign), 1) if benign else 0.0,
                # Youden's J: how much better than chance this threshold
                # separates the two populations. Maximising it picks the
                # point furthest from the no-skill diagonal.
                "youden_j": round(
                    (tp / len(attacks) if attacks else 0)
                    - (fp / len(benign) if benign else 0),
                    4,
                ),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Tune the semantic threshold")
    ap.add_argument("--json", dest="json_out", default=None)
    args = ap.parse_args()

    settings = Settings.from_env()
    embedder = build_embedder(settings)
    if not getattr(embedder, "semantic", False):
        raise SystemExit(
            "Running on the non-semantic hashing stub, so these numbers would\n"
            "mean nothing. Set GEMINI_API_KEY in .env and try again."
        )

    print(f"embedder: {embedder.name} ({getattr(embedder, 'model', '?')})\n")
    defense = SemanticDefense(embedder)
    scored = score_corpus(defense)

    attacks = sorted((s.peak for s in scored if s.kind == "attack"))
    benign = sorted((s.peak for s in scored if s.kind == "benign"))

    def pct(vals, p):
        return vals[min(int(len(vals) * p), len(vals) - 1)] if vals else 0.0

    print("\n\nSeparation between the two populations:\n")
    print("| Population | n | min | p25 | median | p75 | max |")
    print("|---|---|---|---|---|---|---|")
    for name, vals in (("attacks", attacks), ("benign", benign)):
        print(
            f"| {name} | {len(vals)} | {vals[0]:.3f} | {pct(vals,0.25):.3f} | "
            f"{pct(vals,0.5):.3f} | {pct(vals,0.75):.3f} | {vals[-1]:.3f} |"
        )

    gap = benign[-1] - attacks[0] if attacks and benign else 0.0
    print(
        f"\nLowest attack {attacks[0]:.3f} vs highest benign {benign[-1]:.3f} "
        f"-> {'OVERLAP of ' + format(gap, '.3f') if gap > 0 else 'clean separation'}"
    )
    if gap > 0:
        print("Overlap means no threshold separates them perfectly; the sweep")
        print("below shows what each choice costs.")

    rows = sweep(scored)
    best = max(rows, key=lambda r: r["youden_j"])

    print("\n\nThreshold sweep:\n")
    print("| Threshold | Attacks caught | Benign blocked | Youden J |")
    print("|---|---|---|---|")
    for r in rows:
        if r["threshold"] % 0.02 > 0.001 and r["threshold"] != best["threshold"]:
            continue
        mark = "  <- best" if r["threshold"] == best["threshold"] else ""
        print(
            f"| {r['threshold']:.2f} | {r['attacks_caught']}/{r['attacks_total']} "
            f"({r['recall_pct']}%) | {r['benign_blocked']}/{r['benign_total']} "
            f"({r['fpr_pct']}%) | {r['youden_j']}{mark} |"
        )

    current = Settings.from_env()
    print(
        f"\nBest separation at {best['threshold']:.2f}: "
        f"{best['attacks_caught']}/{best['attacks_total']} attacks, "
        f"{best['benign_blocked']}/{best['benign_total']} false positives."
    )
    print("\nTuned on the self-authored corpus - a design parameter, not a")
    print("result. Re-run the held-out set after changing it:")
    print("  python -m bench.heldout.run_heldout --offset 200")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "embedder": embedder.name,
                    "embedding_model": getattr(embedder, "model", None),
                    "cases": [vars(s) for s in scored],
                    "sweep": rows,
                    "best": best,
                },
                indent=2,
            )
        )
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
