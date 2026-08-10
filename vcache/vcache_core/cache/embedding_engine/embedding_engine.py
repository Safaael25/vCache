from abc import ABC, abstractmethod
from typing import Callable, List, Optional


class EmbeddingEngine(ABC):
    """
    Abstract base class for embedding engines.
    """

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """
        Get the embedding for the given text.

        Args:
            text: The text to get the embedding for.

        Returns:
            The embedding of the text as a list of floats.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a batch of texts.

        The default implementation simply loops over `get_embedding`, so every
        existing subclass keeps working without any change. Subclasses backed by
        a model or API that supports true batched execution (e.g. a local
        transformer model, or an API that accepts a list of inputs) should
        override this method, since encoding many texts in one call is
        substantially more efficient than one call per text.

        Args:
            texts: The texts to get embeddings for.

        Returns:
            A list of embeddings, one per input text, in the same order.
        """
        return [self.get_embedding(text) for text in texts]

    def get_engine_factory(self) -> Optional[Callable[[], "EmbeddingEngine"]]:
        """
        Return a zero-argument, picklable factory that reconstructs an
        independent, equivalent instance of this engine.

        This is used by `ConcurrentEmbeddingEngine` to run this engine's
        embedding computation inside separate worker processes, which is only
        beneficial for CPU-bound, in-process models where sidestepping the GIL
        provides real parallelism. Engines that wrap a lightweight remote API
        client are I/O-bound and gain nothing from multiprocessing, so they
        should keep the default `None` return value, which signals that
        process-based execution is not supported for this engine.

        Returns:
            A callable that returns a new instance of this engine, or `None`
            if this engine does not support process-based execution.
        """
        return None
