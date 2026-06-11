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

from atlas_anno.review.label_studio_api import import_batch_tasks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Import Atlas review tasks into a Label Studio project")
    parser.add_argument("--batch", default="pilot_100")
    parser.add_argument("--project-id", default=None, help="Label Studio project ID (auto-resolved if omitted)")
    parser.add_argument("--token", default=None, help="LABEL_STUDIO_TOKEN override (default: read from .env / env var)")
    parser.add_argument("--url", default=None, help="LABEL_STUDIO_URL override (default: read from .env / env var)")
    args = parser.parse_args()

    payload = import_batch_tasks(
        args.batch,
        args.project_id,
        token=args.token,
        url=args.url,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
