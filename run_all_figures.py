#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    package_dir = Path(__file__).resolve().parent
    script = package_dir / "scripts" / "plot_figures.py"
    output_dir = package_dir / "output"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--figure",
            "all",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
