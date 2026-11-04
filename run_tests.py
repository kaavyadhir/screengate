#!/usr/bin/env python3
"""Minimal stdlib test runner.

The suite is plain `test_*` functions with asserts, so `pytest` runs it
natively. This fallback exists so the tests also run in an environment
without pytest installed. Same tests, same assertions.

    python run_tests.py     # fallback
    pytest -q               # preferred, when available
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTS = ROOT / "tests"


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    sys.path.insert(0, str(ROOT))
    passed, failures = 0, []

    for path in sorted(TESTS.glob("test_*.py")):
        module = load(path)
        for name in sorted(n for n in dir(module) if n.startswith("test_")):
            fn = getattr(module, name)
            if not callable(fn):
                continue
            try:
                fn()
                passed += 1
                print(f"  PASS  {path.stem}::{name}")
            except Exception:
                failures.append((f"{path.stem}::{name}", traceback.format_exc()))
                print(f"  FAIL  {path.stem}::{name}")

    print(f"\n{passed} passed, {len(failures)} failed")
    for name, tb in failures:
        print(f"\n--- {name} ---\n{tb}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
