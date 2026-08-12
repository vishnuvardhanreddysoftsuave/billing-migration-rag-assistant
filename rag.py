#!/usr/bin/env python
"""Launcher so the CLI runs from a clone with no install step.

    python rag.py ingest
    python rag.py ask "What does ERR-4032 mean?"
"""

from __future__ import annotations

import sys
from pathlib import Path

# Article text contains non-ASCII characters (→, "smart quotes"); Windows consoles
# often default stdout to cp1252, which raises UnicodeEncodeError on print(). Force
# UTF-8 so `ask`/`search` output never crashes on the terminal it's printed to.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ragchat.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
