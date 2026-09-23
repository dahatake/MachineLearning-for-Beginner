"""Validate local Markdown image and link targets in the setup documentation."""

from __future__ import annotations

import re
import sys
from pathlib import Path


LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    files = [root / "README.md", root / "SETUP.md"]
    errors: list[str] = []
    for document in files:
        for target in LINK.findall(document.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            path = target.split("#", 1)[0].lstrip("/")
            if path and not (root / path).exists():
                errors.append(f"{document.relative_to(root)}: missing target {target}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print("Documentation links are valid.")


if __name__ == "__main__":
    main()
