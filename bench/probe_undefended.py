"""Show what the target model ACTUALLY says when nothing is defending it.

WHY THIS EXISTS
---------------
The ablation against `gemini-flash-latest` reported 0/33 attacks succeeding
in the undefended configuration. That number has two completely different
explanations and the benchmark cannot tell them apart:

  1. The attacks genuinely fail. A current frontier-ish model ignores
     instructions embedded in a document it was asked to summarise, so
     there was never any harm to block.

  2. The harm detectors never saw the output. bench/harm.py opens with
     `if not output: return []`, so an empty or unparseable response
     scores as "no harm" - identical, in the results table, to a model
     that resisted perfectly.

Explanation 2 would make every downstream number meaningless while
looking healthy, which is the same failure mode as the embedding
batching bug. So this prints the raw bytes the model returned and what
each detector made of them, for a handful of cases, and lets a human
decide which explanation is true.

It is a diagnostic, not a benchmark. Nothing it prints goes in the README.

Usage:
    python -m bench.probe_undefended                 # 3 cases
    python -m bench.probe_undefended --limit 6
    python -m bench.probe_undefended --case direct_override_01
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

from bench.harm import DETECTORS, detect
from bench.runner import load_corpus
from screengate.config import LayerToggles, Policy, Settings
from screengate.models import GatewayRequest, Segment, Trust
from screengate.pipeline import Gateway
from screengate.providers import build_provider
from screengate.rag.embeddings import build_embedder


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--provider", default="gemini", choices=["mock", "gemini"])
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--case", action="append", default=[], help="specific case id(s)")
    args = ap.parse_args(argv)

    cases = [c for c in load_corpus() if c.kind == "attack"]
    if args.case:
        wanted = set(args.case)
        cases = [c for c in cases if c.id in wanted]
        missing = wanted - {c.id for c in cases}
        if missing:
            raise SystemExit(f"No such case id(s): {', '.join(sorted(missing))}")
    else:
        # Spread across categories rather than taking the first N, which
        # would all be direct_override and tell us the least.
        seen: set[str] = set()
        spread = []
        for c in cases:
            if c.category not in seen:
                seen.add(c.category)
                spread.append(c)
        cases = spread[: args.limit]

    settings = dataclasses.replace(Settings.from_env(), provider=args.provider)
    # Embeddings are irrelevant here - no layers run - and the hashing
    # stub keeps this diagnostic off the embedding quota entirely.
    settings = dataclasses.replace(settings, embedder="hashing")
    gateway = Gateway(
        provider=build_provider(settings),
        settings=settings,
        embedder=build_embedder(settings),
    )
    undefended = Policy().with_layers(LayerToggles.none())

    print(f"Probing {len(cases)} attack case(s) against {args.provider}, ALL LAYERS OFF.")
    print(f"That is {len(cases)} model calls.\n")

    for c in cases:
        req = GatewayRequest(
            user_input=c.user_input,
            documents=[
                Segment(trust=Trust.DOCUMENT, content=d.content, label=d.label)
                for d in c.documents
            ],
        )
        res = gateway.handle(req, policy=undefended)
        out = res.output

        print("=" * 72)
        print(f"CASE      {c.id}   [{c.category}]")
        print(f"EXPECTED  harm: {', '.join(c.harm) or '(none declared)'}")
        print(f"DECISION  {res.decision.value}")
        print("-" * 72)
        if out is None:
            print("RAW OUTPUT: None  <-- harm detection is BLIND here.")
            print("            Every detector short-circuits on falsy output, so")
            print("            this case scores as 'attack failed' without any")
            print("            evidence that it did. This is the artifact case.")
        elif not out.strip():
            print(f"RAW OUTPUT: empty string ({len(out)} chars of whitespace)")
            print("            Same blind spot as None.")
        else:
            print(f"RAW OUTPUT ({len(out)} chars):")
            print(out if len(out) <= 1200 else out[:1200] + "\n  ...[truncated]")
        print("-" * 72)

        if out:
            policy = Policy()
            fired = detect(out, list(DETECTORS), policy)   # ALL detectors, not just expected
            expected_fired = detect(out, c.harm, policy)
            print(f"DETECTORS firing (all):      {fired or '(none)'}")
            print(f"DETECTORS firing (expected): {expected_fired or '(none)'}")
            if not fired:
                print("VERDICT   Model produced clean output. The attack genuinely")
                print("          did not land - there was nothing to block.")
            else:
                print("VERDICT   Attack LANDED undefended. This case is a valid")
                print("          test of whether the gateway stops it.")
        print()

    print("=" * 72)
    print(
        "Read this before trusting any block rate: a gateway can only be\n"
        "credited with stopping attacks that work without it."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
