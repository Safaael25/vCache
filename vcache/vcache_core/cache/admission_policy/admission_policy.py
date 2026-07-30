from abc import ABC, abstractmethod
from typing import List


class AdmissionPolicy(ABC):
    """
    Abstract base class defining the interface for cache admission policies.

    This class provides a standardized framework for deciding whether a
    cache-miss item is actually worth caching, before it takes up a slot.
    Unlike an `EvictionPolicy` (which decides what to remove once the cache
    is full), an `AdmissionPolicy` decides what gets let in, in the first
    place.
    """

    @abstractmethod
    def should_admit(self, embedding: List[float]) -> bool:
        """
        Decides whether a cache-miss item should actually be added to the
        cache.

        Args:
            embedding: The embedding vector of the prompt that just missed.

        Returns:
            True if the item should be cached, False if it should be
            skipped this time.
        """
        pass
