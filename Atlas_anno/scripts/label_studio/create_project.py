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

from atlas_anno.review.label_studio_api import create_project_from_batch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or update a Label Studio project for an Atlas batch")
    parser.add_argument("--batch", default="pilot_100")
    parser.add_argument("--title", default="Atlas pilot_100 review")
    parser.add_argument("--project-id", default=None)
    args = parser.parse_args()

    payload = create_project_from_batch(args.batch, args.title, project_id=args.project_id)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
