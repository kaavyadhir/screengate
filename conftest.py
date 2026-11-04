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

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
