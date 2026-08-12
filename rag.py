#!/usr/bin/env python
"""Launcher so the CLI runs from a clone with no install step.

    python rag.py ingest
    python rag.py ask "What does ERR-4032 mean?"
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ragchat.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
