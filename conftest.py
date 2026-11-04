"""Make the repository root importable during test collection.

`python -m bench.runner` works without this because -m puts the current
directory on sys.path. pytest does not, so `import screengate` fails at
collection time unless the package is installed or the root is added here.

A conftest.py at the root is the standard fix and keeps the project
runnable straight from a clone, with no `pip install -e .` step.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def isolated_embed_cache(tmp_path, monkeypatch):
    """Point the embedding cache at a throwaway directory for every test.

    The cache that makes benchmark re-runs affordable is a module-level
    global aimed at a real directory on disk, and it is keyed only by
    model name, kind and text. Tests use fake model names and tiny
    inputs like "a", "b", "c" - so the first run of a test WROTE those
    vectors into the developer's real .embed_cache, and every later run
    READ them back and never called the stubbed API at all.

    That produced a test which passed once and failed forever after,
    with an empty call list and no clue why. Worse, it meant the suite's
    results depended on whether someone had run a benchmark recently.

    Autouse so no test can opt out by forgetting. Resetting the two
    module globals matters as much as the directory: _CACHE is loaded
    lazily and memoised, so a stale in-process dict would survive the
    redirect.
    """
    from screengate.rag import embeddings

    monkeypatch.setattr(embeddings, "_CACHE_DIR", tmp_path / "embed_cache")
    monkeypatch.setattr(embeddings, "_CACHE", None)
    monkeypatch.setattr(embeddings, "_CACHE_PATH", None)
    yield
