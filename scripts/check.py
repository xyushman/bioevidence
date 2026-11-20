"""Run the same quality gate as ``make check`` on platforms without make."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


COMMANDS: tuple[tuple[str, ...], ...] = (
    ("uv", "run", "--frozen", "ruff", "check", "src", "tests"),
    ("uv", "run", "--frozen", "ruff", "format", "--check", "src", "tests"),
    ("uv", "run", "--frozen", "mypy", "src"),
    ("uv", "run", "--frozen", "pytest"),
)


def main() -> int:
    """Run every check in order and stop on the first failure."""
    root = Path(__file__).resolve().parents[1]
    for command in COMMANDS:
        print(f"\n$ {' '.join(command)}", flush=True)
        result = subprocess.run(command, cwd=root, check=False)
        if result.returncode != 0:
            print(f"FAILED: {' '.join(command)}", file=sys.stderr)
            return result.returncode
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
