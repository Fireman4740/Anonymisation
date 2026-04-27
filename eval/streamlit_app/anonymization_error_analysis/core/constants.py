from __future__ import annotations

from typing import Dict

SOURCE_LABELS: Dict[str, str] = {
    "benchmark": "Lancer Benchmark",
    "history": "Historique & Comparaison",
    "ablation": "Etudes d'ablation",
}

SOURCE_ORDER = ["benchmark", "history", "ablation"]
DEFAULT_SOURCE = "benchmark"

DATASET_KINDS = ["TAB", "DB-bio", "JSON", "cleanconll2003"]
DETECTION_MODES = ["serial", "parallel"]

RATBENCH_LANGUAGES = ["english", "mandarin", "spanish"]
RATBENCH_LEVELS = ["Tous", "1", "2", "3"]
