from collections import deque
from typing import Deque, List, Tuple

import numpy as np

from vcache.vcache_core.cache.admission_policy.admission_policy import AdmissionPolicy


class SimilarityLFUAdmissionPolicy(AdmissionPolicy):
    """
    A similarity-aware admission policy, adapted from the idea behind
    TinyLFU (used e.g. by the Caffeine cache library).

    TinyLFU decides whether to admit a new key by checking how often that
    *exact* key has recently been probed, using a hashed frequency sketch.
    That doesn't transfer to vCache directly: two semantically identical
    queries with different wording hash completely differently, so an
    exact-key frequency sketch can't see the relationship that vCache's
    whole caching strategy is built around.

    This policy keeps the same underlying idea -- don't admit something
    until it has shown some sign of recurring -- but checks recurrence by
    semantic similarity instead of exact-key hashing, reusing the same
    kind of embedding comparison vCache already relies on elsewhere.

    How it works: cache-miss items are not admitted the first time they're
    seen. Instead, their embedding is kept on a small, bounded "probation"
    list. If a *similar* embedding is seen again while the original is
    still on probation, the item is promoted (admitted) on this second
    sighting and removed from probation. If the probation list is full
    when a new, non-matching item arrives, the oldest probation entry is
    dropped to make room (a FIFO doorkeeper, in TinyLFU's terms).

    Note:
        The probation list is a separate, bounded, in-memory structure
        distinct from the main cache -- it never touches the eviction
        policy or the main vector index, so it does not consume any of the
        main cache's capacity.
    """

    def __init__(
        self,
        admission_similarity_threshold: float = 0.9,
        probation_capacity: int = 1000,
    ):
        """
        Initializes the similarity-aware admission policy.

        Args:
            admission_similarity_threshold: Cosine similarity, in [0, 1],
                above which a new embedding is considered a repeat of an
                existing probation entry and gets promoted (admitted).
                Higher values require a closer match before admitting.
            probation_capacity: The maximum number of not-yet-admitted
                embeddings tracked at once. When full, the oldest entry is
                dropped to make room for a new one.
        """
        if not (0.0 <= admission_similarity_threshold <= 1.0):
            admission_similarity_threshold = 0.9
        if probation_capacity <= 0:
            probation_capacity = 1000

        self.admission_similarity_threshold: float = admission_similarity_threshold
        self.probation_capacity: int = probation_capacity
        self._probation: Deque[Tuple[np.ndarray, np.ndarray]] = deque()

    def should_admit(self, embedding: List[float]) -> bool:
        """
        Decides whether a cache-miss item should actually be added to the
        cache, based on whether a similar item was recently seen while on
        probation.

        Args:
            embedding: The embedding vector of the prompt that just missed.

        Returns:
            True if a similar embedding was already on probation (this is
            treated as a repeat and promoted); False if this is the first
            sighting (the embedding is added to probation instead).
        """
        candidate = np.asarray(embedding, dtype=np.float64)
        candidate_norm = np.linalg.norm(candidate)
        if candidate_norm == 0:
            return False

        for index, (probation_vector, probation_norm) in enumerate(self._probation):
            if probation_norm == 0:
                continue
            similarity = float(
                np.dot(candidate, probation_vector) / (candidate_norm * probation_norm)
            )
            if similarity >= self.admission_similarity_threshold:
                del self._probation[index]
                return True

        if len(self._probation) >= self.probation_capacity:
            self._probation.popleft()
        self._probation.append((candidate, candidate_norm))
        return False

    def __str__(self) -> str:
        """Returns a string representation of the SimilarityLFUAdmissionPolicy.

        Returns:
            str: A string representation of the instance.
        """
        return (
            f"SimilarityLFUAdmissionPolicy("
            f"admission_similarity_threshold={self.admission_similarity_threshold}, "
            f"probation_capacity={self.probation_capacity})"
        )
