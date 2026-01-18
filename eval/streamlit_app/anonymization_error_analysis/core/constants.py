from __future__ import annotations

from typing import Dict

SOURCE_LABELS: Dict[str, str] = {
    "existing_report": "Rapports existants",
    "local_eval": "Evaluation PipeGraph (local)",
    "saved_runs": "Runs sauvegardes",
}

SOURCE_ORDER = ["existing_report", "local_eval", "saved_runs"]
DEFAULT_SOURCE = "existing_report"

DATASET_KINDS = ["TAB", "JSON", "DB-bio"]
DETECTION_MODES = ["serial", "parallel"]
