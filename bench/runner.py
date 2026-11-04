"""Benchmark harness with per-layer ablation.

Produces the numbers in the README. Four things it does that a naive
harness would not:

  1. Runs the corpus undefended first, so every "blocked" attack can be
     checked against whether it would actually have worked. A gateway
     should not get credit for blocking an attack the app shrugs off.

  2. Scores attacks and benign traffic in the same run and reports both.
     Block rate in isolation is not a result - a wall scores 100%.

  3. Excludes cases marked `requires: real_model` from mock-target rates,
     rather than letting the simulation's inability to model paraphrase
     silently score them as "attack failed".

  4. Refuses to report semantic-layer figures when running on a
     non-semantic embedder, instead of publishing a number that looks like
     paraphrase detection and is not.

Usage:
    python -m bench.runner                          # mock target
    python -m bench.runner --provider gemini        # real model
    python -m bench.runner --json results.json
    python -m bench.runner --sweep                  # semantic threshold sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from screengate.config import LayerToggles, Policy, Settings
from screengate.models import Decision, GatewayRequest, Segment, Trust
from screengate.pipeline import Gateway
from screengate.providers import build_provider
from screengate.rag.embeddings import build_embedder

from .harm import detect

CORPUS_DIR = Path(__file__).parent / "corpus"


# ---------------------------------------------------------------- loading


@dataclass
class Case:
    id: str
    kind: str  # "attack" | "benign"
    user_input: str
    documents: list[Segment]
    category: str | None = None
    harm: list[str] = field(default_factory=list)
    tripwire: str | None = None
    requires: str | None = None

    def to_request(self) -> GatewayRequest:
        return GatewayRequest(
            user_input=self.user_input, documents=self.documents, session_id=self.id
        )


def _documents(raw: list[dict[str, Any]] | None) -> list[Segment]:
    docs: list[Segment] = []
    for d in raw or []:
        if "content_repeat" in d:
            spec = d["content_repeat"]
            content = (spec["text"].strip() + "\n\n") * int(spec["times"])
        else:
            content = d.get("content", "")
        docs.append(Segment(trust=Trust.DOCUMENT, content=content, label=d.get("label")))
    return docs


def load_corpus() -> list[Case]:
    # encoding="utf-8" is load-bearing, not tidiness. Path.read_text()
    # defaults to the platform encoding, which is cp1252 on Windows: the
    # obfuscation payloads then decode as mojibake (a zero-width space
    # becomes three visible Latin characters) and silently stop being the
    # attack they were written as. That produced a different block rate on
    # Windows than on Linux from identical code.
    cases: list[Case] = []
    for raw in yaml.safe_load((CORPUS_DIR / "attacks.yaml").read_text(encoding="utf-8")) or []:
        cases.append(
            Case(
                id=raw["id"],
                kind="attack",
                user_input=raw["user_input"].strip(),
                documents=_documents(raw.get("documents")),
                category=raw.get("category"),
                harm=list(raw.get("harm", [])),
                requires=raw.get("requires"),
            )
        )
    for raw in yaml.safe_load((CORPUS_DIR / "benign.yaml").read_text(encoding="utf-8")) or []:
        cases.append(
            Case(
                id=raw["id"],
                kind="benign",
                user_input=raw["user_input"].strip(),
                documents=_documents(raw.get("documents")),
                tripwire=raw.get("tripwire"),
            )
        )
    return cases


# --------------------------------------------------------------- scoring


@dataclass
class CaseResult:
    case_id: str
    kind: str
    category: str | None
    tripwire: str | None
    decision: str
    harm_leaked: list[str]
    rules_fired: list[str]
    excluded: bool = False


@dataclass
class ConfigResult:
    name: str
    layers: list[str]
    results: list[CaseResult]

    @property
    def attacks(self) -> list[CaseResult]:
        return [r for r in self.results if r.kind == "attack" and not r.excluded]

    @property
    def benign(self) -> list[CaseResult]:
        return [r for r in self.results if r.kind == "benign"]

    @property
    def attacks_blocked(self) -> int:
        return sum(1 for r in self.attacks if r.decision == "block")

    @property
    def attacks_leaked(self) -> int:
        return sum(1 for r in self.attacks if r.harm_leaked)

    @property
    def benign_blocked(self) -> int:
        return sum(1 for r in self.benign if r.decision == "block")

    @property
    def block_rate(self) -> float:
        return _pct(self.attacks_blocked, len(self.attacks))

    @property
    def leak_rate(self) -> float:
        return _pct(self.attacks_leaked, len(self.attacks))

    @property
    def benign_pass_rate(self) -> float:
        n = len(self.benign)
        return _pct(n - self.benign_blocked, n)

    @property
    def false_positive_rate(self) -> float:
        return _pct(self.benign_blocked, len(self.benign))


def _pct(num: int, den: int) -> float:
    return round(100.0 * num / den, 1) if den else 0.0


# ---------------------------------------------------------------- running

CONFIGS: list[tuple[str, LayerToggles]] = [
    ("undefended", LayerToggles.none()),
    ("L1 patterns only", LayerToggles.only("inbound_patterns")),
    ("L2 spotlighting only", LayerToggles.only("spotlighting")),
    ("L3 outbound only", LayerToggles.only("outbound_policy")),
    ("L4 semantic only", LayerToggles.only("semantic")),
    ("L1+L2", LayerToggles.only("inbound_patterns", "spotlighting")),
    ("L1+L2+L3", LayerToggles.only("inbound_patterns", "spotlighting", "outbound_policy")),
    ("all four", LayerToggles()),
]


def run_config(
    gateway: Gateway,
    cases: list[Case],
    name: str,
    layers: LayerToggles,
    *,
    real_model: bool,
) -> ConfigResult:
    policy = Policy().with_layers(layers)
    results: list[CaseResult] = []

    for case in cases:
        excluded = case.requires == "real_model" and not real_model
        if excluded:
            results.append(
                CaseResult(
                    case_id=case.id, kind=case.kind, category=case.category,
                    tripwire=case.tripwire, decision="skipped",
                    harm_leaked=[], rules_fired=[], excluded=True,
                )
            )
            continue

        response = gateway.handle(case.to_request(), policy=policy)
        output = None if response.decision is Decision.BLOCK else response.output
        leaked = detect(output, case.harm, policy) if case.kind == "attack" else []
        results.append(
            CaseResult(
                case_id=case.id,
                kind=case.kind,
                category=case.category,
                tripwire=case.tripwire,
                decision=response.decision.value,
                harm_leaked=leaked,
                rules_fired=sorted({f.rule for f in response.findings}),
            )
        )

    return ConfigResult(name=name, layers=layers.enabled_names, results=results)


# -------------------------------------------------------------- reporting


def summary_table(configs: list[ConfigResult], semantic_ok: bool) -> str:
    lines = [
        "| Configuration | Attacks blocked | Attack success | Benign passed | False positives |",
        "|---|---|---|---|---|",
    ]
    for c in configs:
        note = ""
        if "semantic" in c.layers and not semantic_ok:
            note = " *"
        lines.append(
            f"| {c.name}{note} | {c.attacks_blocked}/{len(c.attacks)} ({c.block_rate}%) "
            f"| {c.attacks_leaked}/{len(c.attacks)} ({c.leak_rate}%) "
            f"| {len(c.benign) - c.benign_blocked}/{len(c.benign)} ({c.benign_pass_rate}%) "
            f"| {c.benign_blocked} ({c.false_positive_rate}%) |"
        )
    if not semantic_ok:
        lines.append("")
        lines.append(
            "* Semantic layer ran on the non-semantic hashing embedder. "
            "These rows measure plumbing, not paraphrase detection - set "
            "GEMINI_API_KEY for a real figure."
        )
    return "\n".join(lines)


def category_table(full: ConfigResult, baseline: ConfigResult) -> str:
    by: dict[str, list[CaseResult]] = {}
    for r in full.attacks:
        by.setdefault(r.category or "uncategorised", []).append(r)
    base: dict[str, list[CaseResult]] = {}
    for r in baseline.attacks:
        base.setdefault(r.category or "uncategorised", []).append(r)

    lines = [
        "| Attack category | Cases | Worked undefended | Blocked by gateway |",
        "|---|---|---|---|",
    ]
    for cat in sorted(by):
        rs = by[cat]
        bs = base.get(cat, [])
        lines.append(
            f"| {cat} | {len(rs)} | {sum(1 for r in bs if r.harm_leaked)}/{len(bs)} "
            f"| {sum(1 for r in rs if r.decision == 'block')}/{len(rs)} |"
        )
    return "\n".join(lines)


def false_positive_detail(full: ConfigResult) -> str:
    fps = [r for r in full.benign if r.decision == "block"]
    if not fps:
        return "None. Every benign case passed."
    lines = ["| Case | Tripwire | Rules fired |", "|---|---|---|"]
    for r in fps:
        lines.append(
            f"| `{r.case_id}` | {r.tripwire or '-'} | {', '.join(r.rules_fired)} |"
        )
    return "\n".join(lines)


def threshold_sweep(gateway: Gateway, cases: list[Case], real_model: bool) -> str:
    """Sweep the semantic threshold to show the precision/recall tradeoff."""
    lines = [
        "| Threshold | Attacks blocked | Benign passed |",
        "|---|---|---|",
    ]
    for t in (0.60, 0.65, 0.70, 0.72, 0.75, 0.80, 0.85):
        import dataclasses

        policy = dataclasses.replace(
            Policy(), semantic_threshold=t, layers=LayerToggles.only("semantic")
        )
        atk = ben = atk_blocked = ben_passed = 0
        for case in cases:
            if case.requires == "real_model" and not real_model:
                continue
            r = gateway.handle(case.to_request(), policy=policy)
            blocked = r.decision is Decision.BLOCK
            if case.kind == "attack":
                atk += 1
                atk_blocked += int(blocked)
            else:
                ben += 1
                ben_passed += int(not blocked)
        lines.append(
            f"| {t:.2f} | {atk_blocked}/{atk} ({_pct(atk_blocked, atk)}%) "
            f"| {ben_passed}/{ben} ({_pct(ben_passed, ben)}%) |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ScreenGate benchmark")
    ap.add_argument("--provider", default="mock", choices=["mock", "gemini"])
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--sweep", action="store_true", help="semantic threshold sweep")
    ap.add_argument(
        "--configs",
        default="all",
        help=(
            "Which ablation rows to run: 'all', 'quick' (undefended + full "
            "stack only), or a comma-separated list of row names. A full "
            "ablation against a real model is ~8x the API calls, which is "
            "slow and rate-limited on a free tier; 'quick' still gives the "
            "before/after headline."
        ),
    )
    ap.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="exit non-zero if full-stack block rate falls below this (for CI)",
    )
    args = ap.parse_args(argv)

    import dataclasses

    settings = dataclasses.replace(Settings.from_env(), provider=args.provider)
    embedder = build_embedder(settings)
    gateway = Gateway(
        provider=build_provider(settings), settings=settings, embedder=embedder
    )
    real_model = args.provider == "gemini"
    semantic_ok = gateway.semantic_reportable

    cases = load_corpus()
    skipped = sum(
        1 for c in cases if c.requires == "real_model" and not real_model
    )
    n_atk = sum(1 for c in cases if c.kind == "attack")
    n_ben = len(cases) - n_atk

    print(f"ScreenGate benchmark — target: {args.provider}, embedder: {embedder.name}")
    print(f"{n_atk} attack cases, {n_ben} benign control cases")
    if skipped:
        print(f"{skipped} attack case(s) skipped: require a real model (paraphrase)")
    if not semantic_ok:
        print("Semantic layer: NON-REPORTABLE (hashing embedder, no API key)")
    print()

    if args.configs == "all":
        chosen = CONFIGS
    elif args.configs == "quick":
        chosen = [CONFIGS[0], CONFIGS[-1]]
    else:
        wanted = {c.strip().lower() for c in args.configs.split(",") if c.strip()}
        chosen = [c for c in CONFIGS if c[0].lower() in wanted]
        if not chosen:
            raise SystemExit(
                "No matching config rows. Available: "
                + ", ".join(repr(c[0]) for c in CONFIGS)
            )
        # The baseline is what every other row is measured against, so it
        # is always included even when not explicitly asked for.
        if chosen[0] is not CONFIGS[0]:
            chosen = [CONFIGS[0], *chosen]

    total_calls = len(chosen) * sum(
        1 for c in cases if not (c.requires == "real_model" and not real_model)
    )
    if real_model:
        print(f"~{total_calls} model calls for this run.\n")

    configs = [
        run_config(gateway, cases, n, l, real_model=real_model) for n, l in chosen
    ]
    baseline, full = configs[0], configs[-1]

    print(summary_table(configs, semantic_ok))
    print("\n\nBy attack category (full stack):\n")
    print(category_table(full, baseline))
    print("\n\nFalse positives (full stack):\n")
    print(false_positive_detail(full))

    if args.sweep:
        print("\n\nSemantic threshold sweep:\n")
        print(threshold_sweep(gateway, cases, real_model))

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "provider": args.provider,
                    "embedder": embedder.name,
                    # Recorded because the default chat model is an alias:
                    # a result is only reproducible if you know which
                    # version actually answered.
                    "model": getattr(gateway.provider, "resolved_model", None)
                    or getattr(gateway.provider, "model_name", None),
                    "embedding_model": getattr(embedder, "model", None),
                    "semantic_reportable": semantic_ok,
                    "skipped_requiring_real_model": skipped,
                    "counts": {"attacks": n_atk, "benign": n_ben},
                    "configs": [
                        {
                            "name": c.name,
                            "layers": c.layers,
                            "attacks_blocked": c.attacks_blocked,
                            "attacks_scored": len(c.attacks),
                            "attacks_leaked": c.attacks_leaked,
                            "benign_blocked": c.benign_blocked,
                            "block_rate": c.block_rate,
                            "leak_rate": c.leak_rate,
                            "benign_pass_rate": c.benign_pass_rate,
                            "cases": [vars(r) for r in c.results],
                        }
                        for c in configs
                    ],
                },
                indent=2,
            )
        )
        print(f"\nWrote {args.json_out}")

    if args.fail_under is not None and full.block_rate < args.fail_under:
        print(f"\nFAIL: block rate {full.block_rate}% below {args.fail_under}%")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
