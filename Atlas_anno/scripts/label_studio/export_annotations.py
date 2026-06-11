#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap() -> None:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))


_bootstrap()

from atlas_anno.review.label_studio_api import export_batch_annotations  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Export reviewed annotations from Label Studio")
    parser.add_argument("--batch", default="pilot_100")
    parser.add_argument("--project-id", default=None, help="Label Studio project ID (auto-resolved if omitted)")
    parser.add_argument("--output", default="data/review/pilot_100/label_studio_export.json")
    parser.add_argument("--token", default=None, help="LABEL_STUDIO_TOKEN override (default: read from .env / env var)")
    parser.add_argument("--url", default=None, help="LABEL_STUDIO_URL override (default: read from .env / env var)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    path = export_batch_annotations(
        args.batch,
        args.project_id,
        str(output_path),
        token=args.token,
        url=args.url,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
