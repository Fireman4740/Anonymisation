"""
Thread-safe LRU Model Cache with proper GPU memory cleanup.
Prevents memory leaks when loading/unloading heavy ML models.
"""

import gc
import os
import threading
import logging
from collections import OrderedDict
from typing import Any, Optional, Callable

logger = logging.getLogger("ModelCache")

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False


class ModelCache:
    """
    Thread-safe LRU cache for ML models with automatic eviction and GPU cleanup.

    Features:
    - LRU eviction via OrderedDict (O(1) move_to_end / popitem)
    - Configurable capacity
    - Optional on_evict callback for custom cleanup
    - Automatic GPU memory cleanup on eviction
    - Thread-safe operations
    """

    def __init__(
        self,
        capacity: int = 4,
        on_evict: Optional[Callable[[str, Any], None]] = None,
        name: str = "default",
    ):
        self.capacity = capacity
        self.on_evict = on_evict
        self.name = name
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def put(self, key: str, model: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return

            while len(self._cache) >= self.capacity:
                lru_key, evicted_model = self._cache.popitem(last=False)
                logger.debug(f"[{self.name}] Evicting model: {lru_key}")
                self._cleanup_model(lru_key, evicted_model)

            self._cache[key] = model
            logger.debug(f"[{self.name}] Cached model: {key} (total: {len(self._cache)})")

    def _cleanup_model(self, key: str, model: Any) -> None:
        if self.on_evict:
            try:
                self.on_evict(key, model)
            except Exception as e:
                logger.warning(f"[{self.name}] on_evict callback failed for {key}: {e}")

        try:
            if hasattr(model, "model") and hasattr(model.model, "to"):
                model.model.to("cpu")
            elif hasattr(model, "to"):
                model.to("cpu")
        except Exception:
            pass

        gc.collect()
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def remove(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                model = self._cache.pop(key)
                self._cleanup_model(key, model)
                return model
        return None

    def clear(self) -> None:
        with self._lock:
            while self._cache:
                key, model = self._cache.popitem(last=False)
                self._cleanup_model(key, model)
            logger.info(f"[{self.name}] Cache cleared")

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._cache

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def keys(self) -> list:
        with self._lock:
            return list(self._cache.keys())


# ============ Global Cache Instances ============

_GLINER_CACHE: Optional[ModelCache] = None
_FLAIR_SPACY_CACHE: Optional[ModelCache] = None
_CACHE_LOCK = threading.Lock()


def get_model_cache(cache_type: str = "gliner") -> ModelCache:
    """
    Get or create a global model cache instance.

    Args:
        cache_type: "gliner" or "flair_spacy"

    Returns:
        The ModelCache instance
    """
    global _GLINER_CACHE, _FLAIR_SPACY_CACHE

    with _CACHE_LOCK:
        if cache_type == "gliner":
            if _GLINER_CACHE is None:
                capacity = int(os.getenv("NER_MODEL_CACHE_SIZE", "4"))
                _GLINER_CACHE = ModelCache(capacity=capacity, name="gliner")
            return _GLINER_CACHE

        elif cache_type == "flair_spacy":
            if _FLAIR_SPACY_CACHE is None:
                # Smaller cache for Flair/Spacy (usually 1-2 models)
                _FLAIR_SPACY_CACHE = ModelCache(capacity=2, name="flair_spacy")
            return _FLAIR_SPACY_CACHE

        else:
            raise ValueError(f"Unknown cache type: {cache_type}")


def clear_all_caches() -> None:
    """Clear all global model caches."""
    global _GLINER_CACHE, _FLAIR_SPACY_CACHE

    with _CACHE_LOCK:
        if _GLINER_CACHE:
            _GLINER_CACHE.clear()
        if _FLAIR_SPACY_CACHE:
            _FLAIR_SPACY_CACHE.clear()

    gc.collect()
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
