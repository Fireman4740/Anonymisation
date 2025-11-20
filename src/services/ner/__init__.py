"""
NER (Named Entity Recognition) services with GPU optimization.
"""

from .ensemble import (
    run_gliner,
    merge_ner_lists,
    warm_up_models,
    split_sentences,
    GLINER_ALL_LABELS,
)

try:
    from .gpu_optimizer import (
        create_optimized_pipeline,
        load_gpu_config,
        ParallelNERPipeline,
    )
    GPU_OPTIMIZER_AVAILABLE = True
except ImportError:
    GPU_OPTIMIZER_AVAILABLE = False
    create_optimized_pipeline = None
    load_gpu_config = None
    ParallelNERPipeline = None

__all__ = [
    "run_gliner",
    "merge_ner_lists",
    "warm_up_models",
    "split_sentences",
    "GLINER_ALL_LABELS",
    "create_optimized_pipeline",
    "load_gpu_config",
    "ParallelNERPipeline",
    "GPU_OPTIMIZER_AVAILABLE",
]
