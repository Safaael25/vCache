import functools
from typing import Callable, List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings

from vcache.vcache_core.cache.embedding_engine.embedding_engine import EmbeddingEngine


class LangChainEmbeddingEngine(EmbeddingEngine):
    """
    LangChain implementation of embedding engine using HuggingFace models.

    This engine runs a local model in-process, so its embedding computation is
    CPU-bound (or GPU-bound) rather than I/O-bound. It supports true batched
    encoding and can be reconstructed inside a separate worker process (see
    `get_engine_factory`), which makes it a good candidate for the
    `ConcurrentEmbeddingEngine` process-pool execution mode.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize a LangChain embedding engine.

        Args:
            model_name: Name of the HuggingFace model to use for embeddings.
        """
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for the provided text using LangChain/HuggingFace.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector.

        Raises:
            Exception: If there's an error getting the embedding.
        """
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            raise Exception(f"Error getting embedding from LangChain: {e}")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a batch of texts in a single forward pass.

        A local transformer model encodes a batch of N texts far more
        efficiently than N separate calls, since the per-call Python/tokenizer
        overhead is paid once and the underlying matrix multiplications are
        computed together.

        Args:
            texts: The texts to embed.

        Returns:
            A list of embedding vectors, one per input text, in the same order.

        Raises:
            Exception: If there's an error getting the embeddings.
        """
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            raise Exception(f"Error getting embeddings from LangChain: {e}")

    def get_engine_factory(self) -> Optional[Callable[[], "EmbeddingEngine"]]:
        """
        Return a picklable factory that rebuilds an equivalent engine.

        The underlying HuggingFace model itself is not (efficiently) picklable,
        but the `model_name` needed to reload it is. A worker process can call
        this factory once to load its own copy of the model and reuse it for
        every batch it is asked to embed.

        Returns:
            A `functools.partial` that reconstructs this engine from its
            `model_name` when called with no arguments.
        """
        return functools.partial(LangChainEmbeddingEngine, model_name=self.model_name)
