"""
Thread-safe LRU Model Cache with proper GPU memory cleanup.
Prevents memory leaks when loading/unloading heavy ML models.
"""

import gc
import os
import threading
import logging
from typing import Dict, Any, Optional, Tuple, Callable

logger = logging.getLogger("ModelCache")

# Try importing torch for GPU cleanup
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
    - LRU eviction policy (least recently used)
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
        """
        Initialize the cache.

        Args:
            capacity: Maximum number of models to keep in cache
            on_evict: Optional callback(name, model) called when a model is evicted
            name: Name for logging purposes
        """
        self.capacity = capacity
        self.on_evict = on_evict
        self.name = name
        self._cache: Dict[str, Any] = {}
        self._access_order: list = []
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """
        Get a model from cache, updating access order.

        Args:
            key: Model identifier

        Returns:
            The cached model or None if not found
        """
        with self._lock:
            if key in self._cache:
                # Update LRU order
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]
        return None

    def put(self, key: str, model: Any) -> None:
        """
        Add a model to the cache, evicting LRU if necessary.

        Args:
            key: Model identifier
            model: The model object to cache
        """
        with self._lock:
            # Already in cache - just update access order
            if key in self._cache:
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                return

            # Evict LRU entries if at capacity
            while len(self._cache) >= self.capacity and self._access_order:
                lru_key = self._access_order.pop(0)
                evicted_model = self._cache.pop(lru_key, None)

                if evicted_model is not None:
                    logger.debug(f"[{self.name}] Evicting model: {lru_key}")
                    self._cleanup_model(lru_key, evicted_model)

            # Add new model
            self._cache[key] = model
            self._access_order.append(key)
            logger.debug(f"[{self.name}] Cached model: {key} (total: {len(self._cache)})")

    def _cleanup_model(self, key: str, model: Any) -> None:
        """
        Clean up an evicted model, freeing GPU memory if applicable.
        """
        # Call custom eviction handler if provided
        if self.on_evict:
            try:
                self.on_evict(key, model)
            except Exception as e:
                logger.warning(f"[{self.name}] on_evict callback failed for {key}: {e}")

        # Try to move model to CPU before deletion (frees GPU memory)
        try:
            if hasattr(model, "model") and hasattr(model.model, "to"):
                model.model.to("cpu")
            elif hasattr(model, "to"):
                model.to("cpu")
        except Exception:
            pass

        # Delete the model reference
        try:
            del model
        except Exception:
            pass

        # Force garbage collection and clear GPU cache
        gc.collect()
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def remove(self, key: str) -> Optional[Any]:
        """
        Remove a specific model from the cache.

        Args:
            key: Model identifier

        Returns:
            The removed model or None
        """
        with self._lock:
            if key in self._cache:
                model = self._cache.pop(key)
                if key in self._access_order:
                    self._access_order.remove(key)
                self._cleanup_model(key, model)
                return model
        return None

    def clear(self) -> None:
        """Clear all models from the cache."""
        with self._lock:
            for key in list(self._cache.keys()):
                model = self._cache.pop(key, None)
                if model is not None:
                    self._cleanup_model(key, model)
            self._access_order.clear()
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
