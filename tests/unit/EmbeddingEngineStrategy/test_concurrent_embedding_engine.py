import functools
import hashlib
import threading
import time
import unittest
from typing import List

import pytest

from vcache.vcache_core.cache.cache import Cache
from vcache.vcache_core.cache.embedding_engine.concurrent_embedding_engine import (
    ConcurrentEmbeddingEngine,
    EmbeddingExecutionMode,
)
from vcache.vcache_core.cache.embedding_engine.embedding_engine import EmbeddingEngine
from vcache.vcache_core.cache.embedding_engine.strategies.benchmark import (
    BenchmarkEmbeddingEngine,
)
from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.strategies.in_memory import (
    InMemoryEmbeddingMetadataStorage,
)
from vcache.vcache_core.cache.embedding_store.embedding_store import EmbeddingStore
from vcache.vcache_core.cache.embedding_store.vector_db.strategies.hnsw_lib import (
    HNSWLibVectorDB,
)
from vcache.vcache_core.cache.eviction_policy.strategies.no_eviction import (
    NoEvictionPolicy,
)

_EMBEDDING_DIM = 8


def _deterministic_embedding(text: str) -> List[float]:
    """Deterministic, hash-based fake embedding: same text -> same vector."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [b / 255.0 for b in digest[:_EMBEDDING_DIM]]


class _FakeLocalEmbeddingEngine(EmbeddingEngine):
    """
    Deterministic, picklable stand-in for a local (CPU-bound) embedding engine.

    Must stay a module-level class (not defined inside a test function) so that
    it can be pickled by name when used with EmbeddingExecutionMode.PROCESS.
    """

    def __init__(self, delay_s: float = 0.0):
        self.delay_s = delay_s

    def get_embedding(self, text: str) -> List[float]:
        if self.delay_s:
            time.sleep(self.delay_s)
        return _deterministic_embedding(text)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if self.delay_s:
            time.sleep(self.delay_s)
        return [_deterministic_embedding(text) for text in texts]

    def get_engine_factory(self):
        return functools.partial(_FakeLocalEmbeddingEngine, delay_s=self.delay_s)


class _CountingEmbeddingEngine(EmbeddingEngine):
    """Wraps another engine and records how it was called; thread-safe."""

    def __init__(self, inner: EmbeddingEngine):
        self.inner = inner
        self._lock = threading.Lock()
        self.single_call_count = 0
        self.batch_call_count = 0
        self.batch_sizes: List[int] = []

    def get_embedding(self, text: str) -> List[float]:
        with self._lock:
            self.single_call_count += 1
        return self.inner.get_embedding(text)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        with self._lock:
            self.batch_call_count += 1
            self.batch_sizes.append(len(texts))
        return self.inner.get_embeddings(texts)


class TestConcurrentEmbeddingEngineSync(unittest.TestCase):
    def test_sync_mode_matches_wrapped_engine_and_adds_no_pooling(self):
        """SYNC mode must be a pure passthrough: identical results, no batching."""
        counting_engine = _CountingEmbeddingEngine(_FakeLocalEmbeddingEngine())
        wrapper = ConcurrentEmbeddingEngine(
            engine=counting_engine, mode=EmbeddingExecutionMode.SYNC
        )

        result = wrapper.get_embedding("hello world")

        self.assertEqual(result, _deterministic_embedding("hello world"))
        self.assertEqual(counting_engine.single_call_count, 1)
        self.assertEqual(counting_engine.batch_call_count, 0)
        wrapper.shutdown()


class TestConcurrentEmbeddingEngineThreadMode(unittest.TestCase):
    def test_thread_mode_correctness_sequential(self):
        engine = ConcurrentEmbeddingEngine(
            engine=_FakeLocalEmbeddingEngine(),
            mode=EmbeddingExecutionMode.THREAD,
            num_workers=2,
            max_batch_size=8,
            batch_timeout_s=0.01,
        )
        try:
            for text in ["a", "b", "c", "a completely different sentence"]:
                self.assertEqual(engine.get_embedding(text), _deterministic_embedding(text))
        finally:
            engine.shutdown()

    def test_thread_mode_deduplicates_concurrent_identical_requests(self):
        """
        Many threads requesting the exact same text concurrently should trigger
        at most a handful of underlying computations, not one per caller.
        """
        counting_engine = _CountingEmbeddingEngine(
            _FakeLocalEmbeddingEngine(delay_s=0.1)
        )
        engine = ConcurrentEmbeddingEngine(
            engine=counting_engine,
            mode=EmbeddingExecutionMode.THREAD,
            num_workers=4,
            max_batch_size=32,
            batch_timeout_s=0.2,
        )
        try:
            num_callers = 25
            text = "the exact same prompt"
            barrier = threading.Barrier(num_callers)
            results = [None] * num_callers

            def call(i):
                barrier.wait()
                results[i] = engine.get_embedding(text)

            threads = [threading.Thread(target=call, args=(i,)) for i in range(num_callers)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            expected = _deterministic_embedding(text)
            for result in results:
                self.assertEqual(result, expected)

            # Deduplication means the identical text was embedded far fewer
            # times than the number of concurrent callers.
            total_underlying_calls = sum(counting_engine.batch_sizes)
            self.assertLess(total_underlying_calls, num_callers)
        finally:
            engine.shutdown()

    def test_thread_mode_batches_concurrent_distinct_requests(self):
        """
        Distinct texts arriving concurrently should be grouped into batches of
        more than one item at least once, proving requests are coalesced
        instead of dispatched one-by-one.
        """
        counting_engine = _CountingEmbeddingEngine(
            _FakeLocalEmbeddingEngine(delay_s=0.05)
        )
        engine = ConcurrentEmbeddingEngine(
            engine=counting_engine,
            mode=EmbeddingExecutionMode.THREAD,
            num_workers=1,
            max_batch_size=16,
            batch_timeout_s=0.1,
        )
        try:
            texts = [f"distinct prompt number {i}" for i in range(10)]
            barrier = threading.Barrier(len(texts))
            results = [None] * len(texts)

            def call(i):
                barrier.wait()
                results[i] = engine.get_embedding(texts[i])

            threads = [threading.Thread(target=call, args=(i,)) for i in range(len(texts))]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            for text, result in zip(texts, results):
                self.assertEqual(result, _deterministic_embedding(text))

            self.assertTrue(
                any(size > 1 for size in counting_engine.batch_sizes),
                f"Expected at least one batch with more than one item, got batch sizes: "
                f"{counting_engine.batch_sizes}",
            )
        finally:
            engine.shutdown()

    def test_thread_mode_is_thread_safe_under_mixed_concurrent_load(self):
        """A larger mixed workload (repeats + uniques) should never corrupt results."""
        engine = ConcurrentEmbeddingEngine(
            engine=_FakeLocalEmbeddingEngine(delay_s=0.01),
            mode=EmbeddingExecutionMode.THREAD,
            num_workers=4,
            max_batch_size=8,
            batch_timeout_s=0.02,
        )
        try:
            requests = [f"prompt {i % 7}" for i in range(60)]  # heavy repetition
            results = [None] * len(requests)

            def call(i):
                results[i] = engine.get_embedding(requests[i])

            threads = [threading.Thread(target=call, args=(i,)) for i in range(len(requests))]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=15)

            for text, result in zip(requests, results):
                self.assertEqual(result, _deterministic_embedding(text))
        finally:
            engine.shutdown()


class TestConcurrentEmbeddingEngineProcessMode(unittest.TestCase):
    def test_process_mode_requires_engine_factory(self):
        engine_without_factory = BenchmarkEmbeddingEngine()
        with self.assertRaises(ValueError):
            ConcurrentEmbeddingEngine(
                engine=engine_without_factory, mode=EmbeddingExecutionMode.PROCESS
            )

    def test_process_mode_correctness(self):
        engine = ConcurrentEmbeddingEngine(
            engine=_FakeLocalEmbeddingEngine(),
            mode=EmbeddingExecutionMode.PROCESS,
            num_workers=2,
            max_batch_size=8,
            batch_timeout_s=0.05,
        )
        try:
            texts = ["alpha", "beta", "gamma", "alpha"]
            results = engine.get_embeddings(texts)
            for text, result in zip(texts, results):
                self.assertEqual(result, _deterministic_embedding(text))
        finally:
            engine.shutdown()

    def test_process_mode_handles_concurrent_callers(self):
        engine = ConcurrentEmbeddingEngine(
            engine=_FakeLocalEmbeddingEngine(delay_s=0.02),
            mode=EmbeddingExecutionMode.PROCESS,
            num_workers=2,
            max_batch_size=8,
            batch_timeout_s=0.05,
        )
        try:
            texts = [f"process mode prompt {i % 5}" for i in range(15)]
            results = [None] * len(texts)

            def call(i):
                results[i] = engine.get_embedding(texts[i])

            threads = [threading.Thread(target=call, args=(i,)) for i in range(len(texts))]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=20)

            for text, result in zip(texts, results):
                self.assertEqual(result, _deterministic_embedding(text))
        finally:
            engine.shutdown()


class TestEmbeddingEngineBaseDefaults(unittest.TestCase):
    def test_default_get_embeddings_loops_over_get_embedding(self):
        """An engine that only implements get_embedding must still work batched."""
        engine = _FakeLocalEmbeddingEngineNoBatchOverride()
        texts = ["one", "two", "three"]
        results = engine.get_embeddings(texts)
        self.assertEqual(results, [_deterministic_embedding(t) for t in texts])

    def test_default_get_engine_factory_is_none(self):
        engine = BenchmarkEmbeddingEngine()
        self.assertIsNone(engine.get_engine_factory())


class _FakeLocalEmbeddingEngineNoBatchOverride(EmbeddingEngine):
    """Only implements the required abstract method, to exercise base defaults."""

    def get_embedding(self, text: str) -> List[float]:
        return _deterministic_embedding(text)


class TestConcurrentEmbeddingEngineCacheIntegration(unittest.TestCase):
    """
    Verifies ConcurrentEmbeddingEngine is a true drop-in for Cache: multiple
    threads calling Cache.get_knn / Cache.add concurrently (mirroring
    concurrent vcache.infer() calls) must get correct results.
    """

    def test_concurrent_cache_lookups_through_wrapper(self):
        wrapped_engine = ConcurrentEmbeddingEngine(
            engine=_FakeLocalEmbeddingEngine(delay_s=0.01),
            mode=EmbeddingExecutionMode.THREAD,
            num_workers=4,
            max_batch_size=8,
            batch_timeout_s=0.02,
        )
        cache = Cache(
            embedding_store=EmbeddingStore(
                embedding_metadata_storage=InMemoryEmbeddingMetadataStorage(),
                vector_db=HNSWLibVectorDB(),
            ),
            embedding_engine=wrapped_engine,
            eviction_policy=NoEvictionPolicy(),
        )

        try:
            prompts = [f"cached prompt {i}" for i in range(10)]
            embedding_ids = {}
            for prompt in prompts:
                embedding_ids[prompt] = cache.add(
                    prompt=prompt, response=f"response for {prompt}", id_set=-1
                )

            results = {}
            results_lock = threading.Lock()

            def lookup(prompt):
                knn = cache.get_knn(prompt=prompt, k=1)
                with results_lock:
                    results[prompt] = knn[0][1]  # embedding_id of nearest neighbor

            # Query every prompt (with repeats) concurrently.
            query_plan = prompts * 3
            threads = [threading.Thread(target=lookup, args=(p,)) for p in query_plan]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=15)

            for prompt in prompts:
                self.assertEqual(results[prompt], embedding_ids[prompt])
        finally:
            wrapped_engine.shutdown()


if __name__ == "__main__":
    unittest.main()
