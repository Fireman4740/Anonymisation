#!/usr/bin/env python3
"""Thin wrapper around the unified evaluation CLI (eval.cli.main).

    python scripts/evaluate.py run --dataset tab --config configs/evaluation/no_llm.json
    python scripts/evaluate.py run --dataset all --config configs/evaluation/full_llm.json
    python scripts/evaluate.py ablation --dataset tab \
        --ablation-config configs/evaluation/ablations/default.json
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
