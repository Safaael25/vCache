import atexit
import logging
import queue
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from vcache.vcache_core.cache.embedding_engine.embedding_engine import EmbeddingEngine

logger: logging.Logger = logging.getLogger(__name__)

# Populated once per worker process by `_process_worker_init`. Must be a module
# level global (rather than an attribute on some object) because it is looked
# up by `_process_worker_embed_batch`, which is the picklable, top-level
# function actually submitted to the `ProcessPoolExecutor`.
_worker_engine: Optional[EmbeddingEngine] = None


def _process_worker_init(engine_factory: Callable[[], EmbeddingEngine]) -> None:
    """
    Initializes a worker process with its own copy of the embedding engine.

    Runs exactly once per worker process (via `ProcessPoolExecutor`'s
    `initializer`), so the model backing `engine_factory` is loaded a single
    time per process and then reused for every batch that process handles.

    Args:
        engine_factory: A zero-argument callable that builds a fresh
            `EmbeddingEngine` instance, as returned by
            `EmbeddingEngine.get_engine_factory`.
    """
    global _worker_engine
    _worker_engine = engine_factory()


def _process_worker_embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Embeds a batch of texts using this worker process's engine instance.

    Args:
        texts: The texts to embed.

    Returns:
        A list of embedding vectors, one per input text, in the same order.
    """
    if _worker_engine is None:
        raise RuntimeError(
            "Worker process embedding engine was not initialized. This should "
            "not happen when the worker is created via ConcurrentEmbeddingEngine."
        )
    return _worker_engine.get_embeddings(texts)


class EmbeddingExecutionMode(Enum):
    """
    Execution strategy used by `ConcurrentEmbeddingEngine` to compute embeddings.
    """

    SYNC = "sync"
    """Call the wrapped engine directly, on the calling thread. Identical to
    using the wrapped engine without any dispatcher; this is the
    backward-compatible default."""

    THREAD = "thread"
    """Dispatch (batched) embedding calls to a thread pool. Safe for any
    engine, and the right choice for I/O-bound remote API engines, since
    Python releases the GIL while a thread is blocked on network I/O."""

    PROCESS = "process"
    """Dispatch (batched) embedding calls to a pool of persistent worker
    processes, each holding its own copy of the engine. Only supported for
    engines that implement `get_engine_factory`; provides real multi-core
    parallelism for CPU-bound local models, which a thread pool cannot due to
    the GIL."""


class ConcurrentEmbeddingEngine(EmbeddingEngine):
    """
    An `EmbeddingEngine` decorator that adds concurrent, batched execution.

    This wraps any existing `EmbeddingEngine` and preserves its interface, so
    it is a drop-in replacement anywhere an `EmbeddingEngine` is expected
    (e.g. `VCacheConfig(embedding_engine=...)`) - no other vCache component
    needs to change.

    Concurrent callers of `get_embedding` (e.g. multiple threads each calling
    `vcache.infer()`) have their requests coalesced: identical in-flight texts
    share a single computation and a single result instead of each being
    recomputed, and requests arriving close together are grouped into batches
    (bounded by `max_batch_size` / `batch_timeout_s`) and executed as one call
    to the underlying engine, which is far more efficient for local models
    than one call per text.

    In `EmbeddingExecutionMode.PROCESS` mode, batches are executed on a pool of
    persistent worker processes (see `_process_worker_init`), each holding its
    own copy of the model, which sidesteps the GIL entirely and lets embedding
    computation use multiple CPU cores in parallel. This mode requires the
    wrapped engine to implement `get_engine_factory`; engines that don't (e.g.
    remote API engines, which are I/O-bound and gain nothing from separate
    processes) will raise `ValueError` if `PROCESS` mode is requested.

    Note (Windows/`spawn`): scripts that construct a `ConcurrentEmbeddingEngine`
    with `EmbeddingExecutionMode.PROCESS` must do so under an
    `if __name__ == "__main__":` guard, per the standard `multiprocessing`
    requirement on platforms using the `spawn` start method.
    """

    def __init__(
        self,
        engine: EmbeddingEngine,
        mode: EmbeddingExecutionMode = EmbeddingExecutionMode.THREAD,
        num_workers: int = 4,
        max_batch_size: int = 16,
        batch_timeout_s: float = 0.01,
    ):
        """
        Initializes the concurrent embedding dispatcher.

        Args:
            engine: The underlying embedding engine to wrap.
            mode: The execution strategy to use. Defaults to `THREAD`, which is
                safe for any engine. Use `SYNC` to reproduce the exact
                pre-existing, unwrapped behavior.
            num_workers: The number of threads or processes in the pool. Only
                used when `mode` is `THREAD` or `PROCESS`.
            max_batch_size: The maximum number of requests grouped into a
                single batch.
            batch_timeout_s: How long (in seconds) to wait for a batch to fill
                up to `max_batch_size` before dispatching it anyway. Keeps
                latency bounded for low request rates while still batching
                under load.

        Raises:
            ValueError: If `mode` is `PROCESS` but `engine` does not support
                process-based execution (`engine.get_engine_factory()` is
                `None`).
        """
        if mode == EmbeddingExecutionMode.PROCESS and engine.get_engine_factory() is None:
            raise ValueError(
                f"{type(engine).__name__} does not support "
                "EmbeddingExecutionMode.PROCESS because it does not implement "
                "get_engine_factory(). This is expected for I/O-bound remote "
                "API engines, which should use EmbeddingExecutionMode.THREAD "
                "(or SYNC) instead. Local, in-process engines can opt in by "
                "implementing get_engine_factory()."
            )

        self._engine: EmbeddingEngine = engine
        self._mode: EmbeddingExecutionMode = mode
        self._max_batch_size: int = max(1, max_batch_size)
        self._batch_timeout_s: float = max(0.0, batch_timeout_s)

        # Requests waiting to be picked up by the dispatch loop.
        self._queue: "queue.Queue[Tuple[str, Future]]" = queue.Queue()

        # Coalesces concurrent requests for the same text into one Future, so
        # duplicate work is never scheduled twice.
        self._in_flight_lock: threading.Lock = threading.Lock()
        self._in_flight: Dict[str, Future] = {}

        self._thread_pool: Optional[ThreadPoolExecutor] = None
        self._process_pool: Optional[ProcessPoolExecutor] = None
        self._dispatcher_thread: Optional[threading.Thread] = None
        self._shutdown_event: threading.Event = threading.Event()
        self._is_shutdown: bool = False
        self._shutdown_lock: threading.Lock = threading.Lock()

        if self._mode == EmbeddingExecutionMode.THREAD:
            self._thread_pool = ThreadPoolExecutor(max_workers=num_workers)
        elif self._mode == EmbeddingExecutionMode.PROCESS:
            factory = engine.get_engine_factory()
            self._process_pool = ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_process_worker_init,
                initargs=(factory,),
            )

        if self._mode != EmbeddingExecutionMode.SYNC:
            self._dispatcher_thread = threading.Thread(
                target=self._dispatch_loop,
                name="ConcurrentEmbeddingEngine-dispatcher",
                daemon=True,
            )
            self._dispatcher_thread.start()

        atexit.register(self.shutdown)

    def get_embedding(self, text: str) -> List[float]:
        """
        Gets the embedding for a single text, batching/dedup-ing under the hood.

        Thread-safe: safe to call from multiple threads concurrently (e.g. from
        multiple concurrent `vcache.infer()` calls).

        Args:
            text: The text to get the embedding for.

        Returns:
            The embedding of the text as a list of floats.
        """
        if self._mode == EmbeddingExecutionMode.SYNC:
            return self._engine.get_embedding(text)
        return self._get_or_submit(text).result()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Gets embeddings for a batch of texts, batching/dedup-ing under the hood.

        Args:
            texts: The texts to get embeddings for.

        Returns:
            A list of embeddings, one per input text, in the same order.
        """
        if self._mode == EmbeddingExecutionMode.SYNC:
            return self._engine.get_embeddings(texts)
        futures = [self._get_or_submit(text) for text in texts]
        return [future.result() for future in futures]

    def get_engine_factory(self) -> Optional[Callable[[], EmbeddingEngine]]:
        """
        Not supported: a `ConcurrentEmbeddingEngine` manages its own worker
        pools and dispatcher thread, which are not picklable, so it cannot
        itself be reconstructed inside a worker process.

        Returns:
            None, always.
        """
        return None

    def shutdown(self) -> None:
        """
        Shuts down the dispatcher thread and any worker pools gracefully.

        Safe to call multiple times (e.g. once explicitly and once more via
        the `atexit` hook registered in `__init__`).
        """
        with self._shutdown_lock:
            if self._is_shutdown:
                return
            self._is_shutdown = True

        self._shutdown_event.set()
        if self._dispatcher_thread is not None:
            self._dispatcher_thread.join(timeout=5)
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=True)
        if self._process_pool is not None:
            self._process_pool.shutdown(wait=True)

    def __enter__(self) -> "ConcurrentEmbeddingEngine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()

    def _get_or_submit(self, text: str) -> Future:
        """
        Returns the in-flight `Future` for `text`, creating and enqueuing one
        if no request for this exact text is currently in flight.

        This is the request-coalescing/deduplication step: if two callers ask
        for the same text concurrently, only one embedding computation is
        scheduled and both callers await the same `Future`.

        Args:
            text: The text to get (or start) a future embedding for.

        Returns:
            A `Future` that will resolve to the embedding for `text`.
        """
        with self._in_flight_lock:
            existing = self._in_flight.get(text)
            if existing is not None:
                return existing
            future: Future = Future()
            self._in_flight[text] = future
            self._queue.put((text, future))
            return future

    def _dispatch_loop(self) -> None:
        """
        Background loop that groups queued requests into batches and executes
        them on the configured worker pool.

        Runs on its own daemon thread for the lifetime of this instance. Each
        iteration waits for at least one request, then greedily drains more
        requests (up to `max_batch_size`, up to `batch_timeout_s` after the
        first item arrived) before dispatching the batch, so that requests
        arriving close together are batched while a single isolated request is
        never held up longer than `batch_timeout_s`.
        """
        while not self._shutdown_event.is_set():
            try:
                first_item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            batch: List[Tuple[str, Future]] = [first_item]
            deadline = time.monotonic() + self._batch_timeout_s
            while len(batch) < self._max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=remaining))
                except queue.Empty:
                    break

            self._dispatch_batch(batch)

    def _dispatch_batch(self, batch: List[Tuple[str, Future]]) -> None:
        """
        Submits one batch of requests to the configured worker pool.

        Args:
            batch: A list of (text, future) pairs to execute together.
        """
        texts: List[str] = [text for text, _ in batch]
        futures: List[Future] = [future for _, future in batch]

        if self._mode == EmbeddingExecutionMode.THREAD:
            exec_future = self._thread_pool.submit(self._engine.get_embeddings, texts)
        else:
            exec_future = self._process_pool.submit(
                _process_worker_embed_batch, texts
            )

        exec_future.add_done_callback(
            lambda ef, texts=texts, futures=futures: self._on_batch_done(
                ef, texts, futures
            )
        )

    def _on_batch_done(
        self, exec_future: Future, texts: List[str], futures: List[Future]
    ) -> None:
        """
        Propagates a completed batch's result (or exception) to each waiting
        caller and clears the batch's entries from the in-flight dedup map.

        Args:
            exec_future: The completed `Future` from the worker pool.
            texts: The texts that were part of this batch, in submission order.
            futures: The per-caller futures to resolve, matching `texts` by index.
        """
        try:
            results = exec_future.result()
            for text, future, result in zip(texts, futures, results):
                with self._in_flight_lock:
                    if self._in_flight.get(text) is future:
                        del self._in_flight[text]
                if not future.done():
                    future.set_result(result)
        except BaseException as e:  # noqa: BLE001 - propagate to every waiter
            logger.warning(f"Batch embedding computation failed: {e}")
            for text, future in zip(texts, futures):
                with self._in_flight_lock:
                    if self._in_flight.get(text) is future:
                        del self._in_flight[text]
                if not future.done():
                    future.set_exception(e)
