from typing import List, Optional

from openai import OpenAI as OpenAIClient

from vcache.vcache_core.cache.embedding_engine.embedding_engine import EmbeddingEngine


class OpenAIEmbeddingEngine(EmbeddingEngine):
    """
    OpenAI implementation of embedding engine
    """

    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
    ):
        """
        Initialize an OpenAI embedding engine

        Args:
            model_name: Name of the OpenAI embedding model to use
            api_key: Optional API key (if not provided, will use environment variables)
        """
        self.model_name = model_name
        self.api_key = api_key
        self._client = None

    @property
    def client(self) -> OpenAIClient:
        """
        Lazily initialize the OpenAI client only when needed.

        Returns:
            The OpenAI client instance.
        """
        if self._client is None:
            self._client = (
                OpenAIClient(api_key=self.api_key) if self.api_key else OpenAIClient()
            )
        return self._client

    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for the provided text using OpenAI's API

        Args:
            text: The text to embed

        Returns:
            The embedding vector
        """
        try:
            response = self.client.embeddings.create(input=text, model=self.model_name)
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"Error getting embedding from OpenAI: {e}")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a batch of texts using a single OpenAI API call.

        The OpenAI embeddings endpoint natively accepts a list of inputs, so
        batching multiple texts into one request amortizes network round-trip
        latency across all of them. This engine is still I/O-bound and is
        intentionally not eligible for process-based execution (see
        `get_engine_factory` on the base class): the win here is fewer HTTP
        requests, not multiprocessing.

        Args:
            texts: The texts to embed.

        Returns:
            A list of embedding vectors, one per input text, in the same order.
        """
        try:
            response = self.client.embeddings.create(input=texts, model=self.model_name)
            return [item.embedding for item in response.data]
        except Exception as e:
            raise Exception(f"Error getting embeddings from OpenAI: {e}")
