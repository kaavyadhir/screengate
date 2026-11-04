"""Environment diagnostic.

Prints the interpreter's Unicode version and exactly which attack cases
go unblocked under each configuration.

This exists because the obfuscation attacks depend on how the running
Python normalises zero-width, bidi and soft-hyphen characters, and
`unicodedata` tracks whichever Unicode release that interpreter was built
against. Two machines can therefore produce different block rates from
identical code - which matters, because a README must report numbers the
reader's own machine will reproduce.

    python -m bench.diagnose
"""

from __future__ import annotations

import sys
import unicodedata

from screengate.config import LayerToggles, Policy, Settings
from screengate.pipeline import Gateway
from screengate.providers.mock import MockProvider
from screengate.rag.embeddings import HashingEmbedder

from .runner import load_corpus

CONFIGS = [
    ("L1 only", LayerToggles.only("inbound_patterns")),
    ("L3 only", LayerToggles.only("outbound_policy")),
    ("L1+L2+L3", LayerToggles.only("inbound_patterns", "spotlighting", "outbound_policy")),
    ("all four", LayerToggles()),
]


def main() -> int:
    print(f"python          {sys.version.split()[0]}")
    print(f"unicodedata     {unicodedata.unidata_version}")
    print(f"platform        {sys.platform}")

    gateway = Gateway(
        provider=MockProvider(),
        policy=Policy(),
        settings=Settings(provider="mock", embedder="hashing"),
        embedder=HashingEmbedder(),
    )
    cases = [
        c
        for c in load_corpus()
        if c.kind == "attack" and c.requires != "real_model"
    ]

    for label, toggles in CONFIGS:
        policy = Policy().with_layers(toggles)
        unblocked = []
        for case in cases:
            res = gateway.handle(case.to_request(), policy=policy)
            if res.decision.value != "block":
                unblocked.append((case.id, res.decision.value))
        print(f"\n{label}: {len(cases) - len(unblocked)}/{len(cases)} blocked")
        for cid, decision in unblocked:
            print(f"    not blocked: {cid}  ({decision})")

    # Per-character view of the obfuscation payloads, which is where
    # interpreter differences actually show up.
    print("\nObfuscation payload normalisation:")
    for case in cases:
        if not case.id.startswith("obfuscation"):
            continue
        raw = case.documents[0].content
        nfkc = unicodedata.normalize("NFKC", raw)
        hidden = sorted({f"U+{ord(ch):04X}" for ch in raw if ord(ch) > 127})
        survived = sorted({f"U+{ord(ch):04X}" for ch in nfkc if ord(ch) > 127})
        print(f"  {case.id}")
        print(f"    non-ascii in source   : {hidden}")
        print(f"    surviving after NFKC  : {survived}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
