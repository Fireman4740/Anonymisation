#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap() -> None:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))


_bootstrap()

from atlas_anno.review.label_studio_api import sync_batch_reviews  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a Label Studio export into Atlas reviewed annotations")
    parser.add_argument("--batch", default="pilot_100")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    payload = sync_batch_reviews(args.batch, args.input)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
