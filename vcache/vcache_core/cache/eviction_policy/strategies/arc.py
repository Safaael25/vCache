from datetime import datetime, timezone
from typing import List, Set, Tuple

from vcache.vcache_core.cache.embedding_store.embedding_metadata_storage.embedding_metadata_obj import (
    EmbeddingMetadataObj,
)
from vcache.vcache_core.cache.eviction_policy.eviction_policy import EvictionPolicy

_SEED_OBSERVATIONS: int = 2  # every EmbeddingMetadataObj starts with 2 synthetic seed observations


class ARCEvictionPolicy(EvictionPolicy):
    _MIN_DATETIME: datetime = datetime.min.replace(tzinfo=timezone.utc)

    def __init__(
        self,
        max_size: int,
        watermark: float = 0.95,
        eviction_percentage: float = 0.1,
        initial_p: float = 0.5,
        p_step: float = 0.1,
    ):
        """
        ARC-inspired eviction policy (Adaptive Replacement Cache).

        Approximates Megiddo & Modha's ARC algorithm within this codebase's
        batch/watermark eviction model. Textbook ARC needs per-event ghost
        lists (B1/B2: the identities of items recently evicted from the
        recency list T1 and the frequency list T2) so that a *miss* which
        matches a ghost entry can drive self-tuning of `p`, the target size
        of T1. This `EvictionPolicy` interface has no visibility into misses
        or evictions at all -- only hits (`update_eviction_metadata`) and
        periodic batch victim selection (`select_victims`) -- so there is no
        way to detect a true ghost hit here.

        This implementation keeps ARC's core idea -- splitting the cache into
        a recency list T1 (items referenced exactly once) and a frequency
        list T2 (items referenced more than once), and adaptively deciding
        how many victims to draw from each -- but substitutes an observable
        proxy for ghost-list hits: whether items that were in T1 as of the
        previous eviction round have since been re-referenced and promoted
        into T2. Frequent promotions mean the recency list is proving
        valuable, so `p` (the target fraction of victims drawn from T1) is
        increased, shrinking T1's share of evictions -- the same direction
        textbook ARC moves in response to a B1 ghost hit. No promotions
        pushes `p` back down, mirroring the effect of a B2 ghost hit. This is
        an ARC-inspired approximation, not a faithful port of the original
        algorithm.

        Within each list, victims are chosen by plain recency (T1) or by
        access count then recency (T2), matching ARC's internal LRU ordering
        of each list.

        Args:
            max_size: The absolute maximum number of items the cache can hold.
            watermark: The percentage of `max_size` that triggers eviction.
            eviction_percentage: The percentage of `max_size` to evict.
            initial_p: Starting target fraction, in [0, 1], of victims drawn
                from T1 (the recency list) rather than T2 (the frequency
                list).
            p_step: How much `p` moves per eviction round in response to the
                promotion signal, in [0, 1].
        """
        super().__init__(
            max_size=max_size,
            watermark=watermark,
            eviction_percentage=eviction_percentage,
        )

        if not (0 <= initial_p <= 1.0):
            initial_p = 0.5
            self.logger.warning("initial_p must be in [0,1]. Setting to 0.5.")
        if not (0 <= p_step <= 1.0):
            p_step = 0.1
            self.logger.warning("p_step must be in [0,1]. Setting to 0.1.")

        self.p: float = initial_p
        self.p_step: float = p_step
        self._prev_t1_ids: Set[int] = set()

    def update_eviction_metadata(self, metadata: EmbeddingMetadataObj) -> None:
        """Updates the metadata object's last-accessed timestamp.

        Args:
            metadata (EmbeddingMetadataObj): The metadata object to update.
        """
        metadata.last_accessed = datetime.now(timezone.utc)

    def select_victims(self, all_metadata: List[EmbeddingMetadataObj]) -> List[int]:
        """Selects victims by splitting the cache into recency/frequency
        lists and drawing from each according to the adaptive split `p`.

        Args:
            all_metadata (List[EmbeddingMetadataObj]): A list of all metadata
                objects in the cache.

        Returns:
            List[int]: A list of embedding IDs for the items to be evicted.
        """
        num_to_evict: int = int(self.max_size * self.eviction_percentage)
        if num_to_evict == 0 or not all_metadata:
            return []

        t1: List[EmbeddingMetadataObj] = []
        t2: List[EmbeddingMetadataObj] = []
        for meta in all_metadata:
            (t2 if self._hit_count(meta) >= 1 else t1).append(meta)

        self._adapt_p(t1, t2)

        t1_sorted = sorted(t1, key=self._staleness_key)
        t2_sorted = sorted(
            t2, key=lambda m: (self._hit_count(m), self._staleness_key(m))
        )

        n_t1_victims: int = min(len(t1_sorted), round(self.p * num_to_evict))
        n_t2_victims: int = num_to_evict - n_t1_victims
        if n_t2_victims > len(t2_sorted):
            n_t1_victims = min(
                len(t1_sorted), n_t1_victims + (n_t2_victims - len(t2_sorted))
            )
            n_t2_victims = len(t2_sorted)

        victims: List[int] = [m.embedding_id for m in t1_sorted[:n_t1_victims]] + [
            m.embedding_id for m in t2_sorted[:n_t2_victims]
        ]
        return victims[:num_to_evict]

    def _adapt_p(
        self, t1: List[EmbeddingMetadataObj], t2: List[EmbeddingMetadataObj]
    ) -> None:
        """Nudges `p` using promotions from T1 to T2 as a proxy for ARC's
        ghost-list hits.

        Args:
            t1 (List[EmbeddingMetadataObj]): Current recency-list members.
            t2 (List[EmbeddingMetadataObj]): Current frequency-list members.
        """
        t1_ids: Set[int] = {m.embedding_id for m in t1}
        t2_ids: Set[int] = {m.embedding_id for m in t2}
        promoted: Set[int] = self._prev_t1_ids & t2_ids
        if promoted:
            self.p = min(1.0, self.p + self.p_step)
        else:
            self.p = max(0.0, self.p - self.p_step)
        self._prev_t1_ids = t1_ids

    @staticmethod
    def _hit_count(metadata: EmbeddingMetadataObj) -> int:
        """Approximates access frequency from the count of Bayesian
        observations recorded against this item, beyond the two synthetic
        seed observations every item starts with.

        Args:
            metadata (EmbeddingMetadataObj): The metadata object to inspect.

        Returns:
            int: The approximate number of times this item has been
            referenced since insertion.
        """
        return max(0, len(metadata.observations) - _SEED_OBSERVATIONS)

    def _staleness_key(self, metadata: EmbeddingMetadataObj) -> datetime:
        """Returns the sort key used to find the least-recently-used item
        within a list.

        Args:
            metadata (EmbeddingMetadataObj): The metadata object to inspect.

        Returns:
            datetime: `last_accessed`, or the minimum possible datetime for
            items that were never accessed.
        """
        return (
            metadata.last_accessed
            if metadata.last_accessed is not None
            else self._MIN_DATETIME
        )

    def __str__(self) -> str:
        """Returns a string representation of the ARCEvictionPolicy.

        Returns:
            str: A string representation of the instance.
        """
        return (
            f"ARCEvictionPolicy(max_size={self.max_size}, "
            f"watermark={self.watermark}, "
            f"eviction_percentage={self.eviction_percentage}, "
            f"p={self.p:.3f})"
        )
