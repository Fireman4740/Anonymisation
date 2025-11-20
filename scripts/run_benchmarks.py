#!/usr/bin/env python3
"""Unified benchmark runner for anonymisation pipelines."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.benchmarks import main


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
