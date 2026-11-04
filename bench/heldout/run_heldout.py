"""Held-out evaluation against a public injection dataset.

WHY THIS EXISTS
---------------
Every number in bench/runner.py comes from a corpus written by the same
person who wrote the defenses. That makes the headline block rate partly a
measure of "I blocked what I thought of", and no amount of care in writing
the corpus fixes that - the bias is structural.

This script runs the gateway against prompts written by other people, for
other systems, which were never consulted while tuning a single rule or
threshold. It is run ONCE and the result is published whatever it is. If
the figure is 60%, the README says 60%.

Nothing in this file may be used to tune anything. If a rule is changed
after seeing these results, the result is no longer held out and must be
regenerated from a dataset that has not been looked at.

Default dataset: deepset/prompt-injections (HuggingFace), a labelled set of
injection and benign prompts. Any dataset with a text column and a binary
label works - pass --dataset and --split.

Requires network and `pip install datasets`.

Usage:
    python -m bench.heldout.run_heldout
    python -m bench.heldout.run_heldout --provider gemini --limit 200
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path

from screengate.config import LayerToggles, Policy, Settings
from screengate.models import Decision, GatewayRequest, Segment, Trust
from screengate.pipeline import Gateway
from screengate.providers import build_provider
from screengate.rag.embeddings import build_embedder

TEXT_COLUMNS = ("text", "prompt", "input", "sentence")
LABEL_COLUMNS = ("label", "is_injection", "malicious", "target")


def load_rows(
    dataset: str, split: str, limit: int | None, offset: int = 0
) -> list[tuple[str, int]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "The 'datasets' package is required.\n"
            "  pip install datasets\n"
            "This script also needs network access."
        ) from exc

    try:
        ds = load_dataset(dataset, split=split)
    except Exception as exc:
        # The most common failure is a split that does not exist, so say
        # which ones do rather than surfacing the library's traceback.
        try:
            from datasets import get_dataset_split_names

            names = get_dataset_split_names(dataset)
            raise SystemExit(
                f"Could not load split {split!r} of {dataset!r}.\n"
                f"Available splits: {', '.join(names)}\n"
                f"Re-run with --split <name>."
            ) from exc
        except SystemExit:
            raise
        except Exception:
            raise SystemExit(
                f"Could not load {dataset!r}: {exc}\n"
                "Check the dataset name, or that this machine has network access."
            ) from exc

    cols = ds.column_names
    text_col = next((c for c in TEXT_COLUMNS if c in cols), None)
    label_col = next((c for c in LABEL_COLUMNS if c in cols), None)
    if text_col is None or label_col is None:
        raise SystemExit(
            f"Could not find a text and a label column in {cols}.\n"
            f"Looked for text in {TEXT_COLUMNS} and labels in {LABEL_COLUMNS}."
        )
    print(f"columns: text={text_col!r} label={label_col!r} rows={len(ds)}")

    rows: list[tuple[str, int]] = []
    for i, row in enumerate(ds):
        if i < offset:
            continue
        if limit is not None and i >= offset + limit:
            break
        text = (row[text_col] or "").strip()
        if not text:
            continue
        try:
            label = int(row[label_col])
        except (TypeError, ValueError):
            continue
        rows.append((text, 1 if label else 0))
    return rows


# deepset/prompt-injections is bilingual. The pattern rules are English
# only, so a German prompt cannot match them - that is a scope limit, not
# a detection failure, and reporting one blended number hides it. Crude
# but adequate: these markers do not occur in ordinary English.
GERMAN_MARKERS = re.compile(
    r"[äöüßÄÖÜ]|\b(?:der|die|das|und|ist|nicht|du|bist|ich|eine|einen|"
    r"alle|nun|jetzt|bitte|werden|folgende[nr]?|Anweisungen|Aufgabe)\b",
    re.I,
)


def detect_language(text: str) -> str:
    """'de' or 'en', by marker count. Good enough to split a report."""
    return "de" if len(GERMAN_MARKERS.findall(text)) >= 2 else "en"


def as_request(text: str) -> GatewayRequest:
    """Place the dataset prompt where this threat model puts untrusted text.

    These prompts were written as direct user input to other systems. Here
    they are embedded in a resume, because that is the channel this gateway
    defends. The wrapper text is deliberately minimal so the prompt itself
    carries the signal.
    """
    resume = (
        "Candidate — Backend Engineer. 3 yrs Python, PostgreSQL, REST APIs.\n\n"
        f"{text}"
    )
    return GatewayRequest(
        user_input="Screen this candidate against the role.",
        documents=[Segment(trust=Trust.DOCUMENT, content=resume, label="resume.pdf")],
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Held-out injection evaluation")
    ap.add_argument("--dataset", default="deepset/prompt-injections")
    ap.add_argument("--split", default="train")
    ap.add_argument("--provider", default="mock", choices=["mock", "gemini"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--offset",
        type=int,
        default=0,
        help=(
            "Skip this many rows first. Used to keep a clean split: the "
            "first run looked at rows 0-199, so a rule changed afterwards "
            "must be re-measured from --offset 200 onward, on rows nothing "
            "has been tuned against."
        ),
    )
    ap.add_argument("--json", dest="json_out", default="bench/heldout/results.json")
    args = ap.parse_args(argv)

    rows = load_rows(args.dataset, args.split, args.limit, args.offset)
    if not rows:
        raise SystemExit("Dataset returned no usable rows.")

    settings = dataclasses.replace(Settings.from_env(), provider=args.provider)
    embedder = build_embedder(settings)
    gateway = Gateway(
        provider=build_provider(settings), settings=settings, embedder=embedder
    )
    policy = Policy().with_layers(LayerToggles())

    tp = fp = tn = fn = 0
    missed: list[str] = []
    by_lang: dict[str, dict[str, int]] = {}

    for text, label in rows:
        blocked = gateway.handle(as_request(text), policy=policy).decision is Decision.BLOCK
        lang = detect_language(text)
        bucket = by_lang.setdefault(lang, {"tp": 0, "fn": 0, "fp": 0, "tn": 0})
        bucket[("tp" if blocked else "fn") if label == 1 else ("fp" if blocked else "tn")] += 1
        if label == 1:
            if blocked:
                tp += 1
            else:
                fn += 1
                if len(missed) < 15:
                    missed.append(text[:160])
        else:
            if blocked:
                fp += 1
            else:
                tn += 1

    n_atk, n_ben = tp + fn, tn + fp
    recall = 100.0 * tp / n_atk if n_atk else 0.0
    precision = 100.0 * tp / (tp + fp) if (tp + fp) else 0.0
    fpr = 100.0 * fp / n_ben if n_ben else 0.0

    print(f"Held-out evaluation — {args.dataset} [{args.split}] rows {args.offset}–{args.offset + len(rows)}")
    print(f"target: {args.provider}, embedder: {embedder.name}")
    if not gateway.semantic_reportable:
        print("WARNING: non-semantic embedder; the semantic layer contributes nothing here.")
    print(f"\n{n_atk} injection prompts, {n_ben} benign prompts\n")
    print("| Metric | Value |")
    print("|---|---|")
    print(f"| Injections caught (recall) | {tp}/{n_atk} ({recall:.1f}%) |")
    print(f"| Precision | {precision:.1f}% |")
    print(f"| Benign blocked (false positive rate) | {fp}/{n_ben} ({fpr:.1f}%) |")

    if len(by_lang) > 1:
        print("\nBy language — the rules are English-only, so this split\n"
              "separates a scope limit from a detection failure:\n")
        print("| Language | Injections caught | Benign blocked |")
        print("|---|---|---|")
        for lang in sorted(by_lang):
            b = by_lang[lang]
            atk = b["tp"] + b["fn"]
            ben = b["fp"] + b["tn"]
            r = 100 * b["tp"] / atk if atk else 0.0
            f = 100 * b["fp"] / ben if ben else 0.0
            print(f"| {lang} | {b['tp']}/{atk} ({r:.1f}%) | {b['fp']}/{ben} ({f:.1f}%) |")

    if missed:
        print("\nSample of missed injections (these are the honest failures):\n")
        for m in missed:
            print(f"  - {m!r}")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "split": args.split,
                    "provider": args.provider,
                    "embedder": embedder.name,
                    "semantic_reportable": gateway.semantic_reportable,
                    "counts": {"injection": n_atk, "benign": n_ben},
                    "true_positive": tp,
                    "false_negative": fn,
                    "false_positive": fp,
                    "true_negative": tn,
                    "recall_pct": round(recall, 1),
                    "precision_pct": round(precision, 1),
                    "false_positive_rate_pct": round(fpr, 1),
                    "missed_sample": missed,
                    "by_language": by_lang,
                },
                indent=2,
            )
        )
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
