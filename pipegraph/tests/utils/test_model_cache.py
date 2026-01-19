"""
Unit tests for model_cache.py - Thread-safe LRU model cache.
"""

import pytest
import threading
import time
from unittest.mock import Mock, patch

from src.utils.model_cache import ModelCache, get_model_cache, clear_all_caches


class TestModelCache:
    """Tests for the ModelCache class."""

    def test_init_default_capacity(self):
        """Test default initialization."""
        cache = ModelCache()
        assert cache.capacity == 4
        assert cache.name == "default"
        assert len(cache) == 0

    def test_init_custom_capacity(self):
        """Test custom capacity initialization."""
        cache = ModelCache(capacity=10, name="test_cache")
        assert cache.capacity == 10
        assert cache.name == "test_cache"

    def test_put_and_get(self):
        """Test basic put and get operations."""
        cache = ModelCache(capacity=3)
        model = Mock()

        cache.put("model_a", model)
        assert "model_a" in cache
        assert cache.get("model_a") is model

    def test_get_missing_key(self):
        """Test get returns None for missing keys."""
        cache = ModelCache()
        assert cache.get("nonexistent") is None

    def test_lru_eviction(self):
        """Test LRU eviction when capacity is reached."""
        cache = ModelCache(capacity=2)

        model_a = Mock()
        model_b = Mock()
        model_c = Mock()

        cache.put("a", model_a)
        cache.put("b", model_b)

        assert "a" in cache
        assert "b" in cache
        assert len(cache) == 2

        # Adding c should evict a (LRU)
        cache.put("c", model_c)

        assert "a" not in cache
        assert "b" in cache
        assert "c" in cache
        assert len(cache) == 2

    def test_access_order_updates(self):
        """Test that accessing an item updates its LRU position."""
        cache = ModelCache(capacity=2)

        model_a = Mock()
        model_b = Mock()
        model_c = Mock()

        cache.put("a", model_a)
        cache.put("b", model_b)

        # Access 'a' to move it to most-recently-used
        cache.get("a")

        # Adding c should evict b (now LRU, not a)
        cache.put("c", model_c)

        assert "a" in cache
        assert "b" not in cache
        assert "c" in cache

    def test_put_existing_key_updates_order(self):
        """Test that putting an existing key updates access order without duplication."""
        cache = ModelCache(capacity=2)

        model = Mock()
        cache.put("a", model)
        cache.put("b", Mock())

        # Re-put 'a' to refresh its access order
        cache.put("a", model)

        # Adding c should evict b
        cache.put("c", Mock())

        assert "a" in cache
        assert len(cache) == 2

    def test_remove(self):
        """Test explicit removal of an item."""
        cache = ModelCache(capacity=3)
        model = Mock()

        cache.put("a", model)
        assert "a" in cache

        removed = cache.remove("a")
        # Note: removed is the model before cleanup
        assert "a" not in cache

    def test_remove_missing_key(self):
        """Test removing a non-existent key."""
        cache = ModelCache()
        result = cache.remove("nonexistent")
        assert result is None

    def test_clear(self):
        """Test clearing the entire cache."""
        cache = ModelCache(capacity=5)

        for i in range(3):
            cache.put(f"model_{i}", Mock())

        assert len(cache) == 3

        cache.clear()

        assert len(cache) == 0
        assert cache.keys == []

    def test_on_evict_callback(self):
        """Test that on_evict callback is called during eviction."""
        callback = Mock()
        cache = ModelCache(capacity=1, on_evict=callback)

        model_a = Mock()
        model_b = Mock()

        cache.put("a", model_a)
        cache.put("b", model_b)  # Should evict 'a'

        callback.assert_called_once()
        call_args = callback.call_args[0]
        assert call_args[0] == "a"
        assert call_args[1] is model_a

    def test_thread_safety(self):
        """Test thread safety of cache operations."""
        cache = ModelCache(capacity=100)
        errors = []

        def writer(thread_id):
            try:
                for i in range(50):
                    cache.put(f"thread_{thread_id}_model_{i}", Mock())
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader(thread_id):
            try:
                for i in range(50):
                    cache.get(f"thread_{thread_id}_model_{i}")
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=writer, args=(i,)))
            threads.append(threading.Thread(target=reader, args=(i,)))

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_keys_property(self):
        """Test keys property returns current cache keys."""
        cache = ModelCache()

        cache.put("a", Mock())
        cache.put("b", Mock())

        keys = cache.keys
        assert set(keys) == {"a", "b"}

    def test_contains(self):
        """Test __contains__ (in operator)."""
        cache = ModelCache()
        cache.put("present", Mock())

        assert "present" in cache
        assert "absent" not in cache


class TestGlobalCaches:
    """Tests for global cache functions."""

    def test_get_model_cache_gliner(self):
        """Test getting GLiNER cache."""
        cache = get_model_cache("gliner")
        assert cache is not None
        assert cache.name == "gliner"

        # Should return same instance
        cache2 = get_model_cache("gliner")
        assert cache is cache2

    def test_get_model_cache_flair_spacy(self):
        """Test getting Flair/Spacy cache."""
        cache = get_model_cache("flair_spacy")
        assert cache is not None
        assert cache.name == "flair_spacy"

    def test_get_model_cache_invalid_type(self):
        """Test invalid cache type raises error."""
        with pytest.raises(ValueError, match="Unknown cache type"):
            get_model_cache("invalid_type")

    def test_clear_all_caches(self):
        """Test clearing all global caches."""
        # Add something to caches
        gliner_cache = get_model_cache("gliner")
        gliner_cache.put("test_model", Mock())

        flair_cache = get_model_cache("flair_spacy")
        flair_cache.put("test_flair", Mock())

        # Clear all
        clear_all_caches()

        # Caches should be empty
        assert len(gliner_cache) == 0
        assert len(flair_cache) == 0


class TestModelCleanup:
    """Tests for model cleanup (GPU memory)."""

    def test_cleanup_model_with_to_method(self):
        """Test cleanup moves model to CPU."""
        cache = ModelCache(capacity=1)

        # Mock model with 'to' method but NO nested 'model' attribute
        # (cleanup checks model.model first, then model.to)
        model = Mock(spec=["to"])  # No 'model' attribute
        model.to = Mock()

        cache.put("model", model)
        cache.put("another", Mock())  # Evict 'model'

        # Should have called to("cpu")
        model.to.assert_called_with("cpu")

    def test_cleanup_model_with_nested_model_attr(self):
        """Test cleanup handles nested model.model attribute."""
        cache = ModelCache(capacity=1)

        # Mock GLiNER-style model with nested model attribute
        inner_model = Mock()
        inner_model.to = Mock()

        wrapper = Mock()
        wrapper.model = inner_model

        cache.put("gliner", wrapper)
        cache.put("another", Mock())  # Evict 'gliner'

        inner_model.to.assert_called_with("cpu")
