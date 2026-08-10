import time
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from vcache.vcache_core.cache.eviction_policy.strategies.arc import (
    ARCEvictionPolicy,
)
from vcache.vcache_core.cache.eviction_policy.strategies.cost_aware import (
    CostAwareEvictionPolicy,
)
from vcache.vcache_core.cache.eviction_policy.strategies.fifo import (
    FIFOEvictionPolicy,
)
from vcache.vcache_core.cache.eviction_policy.strategies.lru import (
    LRUEvictionPolicy,
)
from vcache.vcache_core.cache.eviction_policy.strategies.mru import (
    MRUEvictionPolicy,
)
from vcache.vcache_core.cache.eviction_policy.strategies.no_eviction import (
    NoEvictionPolicy,
)
from vcache.vcache_core.cache.eviction_policy.strategies.scu import (
    SCUEvictionPolicy,
)


class TestEvictionPolicyStrategies(unittest.TestCase):
    def setUp(self):
        """Set up mock data for testing eviction policies."""
        self.max_size = 10
        self.eviction_percentage = 0.2
        self.num_to_evict = int(self.max_size * self.eviction_percentage)

        self.metadata = []
        for i in range(5):
            mock_meta = MagicMock()
            mock_meta.embedding_id = i
            mock_meta.created_at = datetime.now(timezone.utc)
            mock_meta.last_accessed = mock_meta.created_at
            mock_meta.cost = None
            self.metadata.append(mock_meta)
            time.sleep(0.01)  # Ensure timestamps are distinct

    def test_fifo_eviction(self):
        """Test the FIFO eviction strategy."""
        policy = FIFOEvictionPolicy(self.max_size, 0.9, self.eviction_percentage)
        victims = policy.select_victims(self.metadata)

        # FIFO should evict the first items added (0 and 1)
        self.assertEqual(len(victims), self.num_to_evict)
        self.assertEqual(sorted(victims), [0, 1])

    def test_lru_eviction(self):
        """Test the LRU eviction strategy."""
        policy = LRUEvictionPolicy(self.max_size, 0.9, self.eviction_percentage)

        # Simulate access to items 0 and 1, making them the most recently used
        policy.update_eviction_metadata(self.metadata[0])
        time.sleep(0.01)
        policy.update_eviction_metadata(self.metadata[1])

        victims = policy.select_victims(self.metadata)

        # LRU should evict the least recently used items (2 and 3)
        self.assertEqual(len(victims), self.num_to_evict)
        self.assertEqual(sorted(victims), [2, 3])

    def test_mru_eviction(self):
        """Test the MRU eviction strategy."""
        policy = MRUEvictionPolicy(self.max_size, 0.9, self.eviction_percentage)

        # Simulate access to items 3 and 4, making them the most recently used
        policy.update_eviction_metadata(self.metadata[3])
        time.sleep(0.01)
        policy.update_eviction_metadata(self.metadata[4])

        victims = policy.select_victims(self.metadata)

        # MRU should evict the most recently used items (3 and 4)
        self.assertEqual(len(victims), self.num_to_evict)
        self.assertEqual(sorted(victims), [3, 4])

    def test_no_eviction(self):
        """Test the NoEviction policy."""
        policy = NoEvictionPolicy()
        victims = policy.select_victims(self.metadata)

        # NoEviction policy should never select any victims
        self.assertEqual(len(victims), 0)

    def test_scu_eviction(self):
        """Test the SCU eviction strategy."""
        policy = SCUEvictionPolicy(self.max_size, 0.9, self.eviction_percentage)

        # Manually set t_prime and observations for a controlled test
        # Item 0: Proven Loser (high t_prime, high n_obs) -> High distance
        self.metadata[0].t_prime = 0.9
        self.metadata[0].observations = [0] * 10

        # Item 1: Suspected Loser (high t_prime, low n_obs) -> Mid distance
        self.metadata[1].t_prime = 0.9
        self.metadata[1].observations = [0] * 2

        # Item 2: Proven Winner (low t_prime, high n_obs) -> Low distance
        self.metadata[2].t_prime = 0.1
        self.metadata[2].observations = [0] * 10

        # Item 3: Suspected Winner (low t_prime, low n_obs) -> Mid-low distance
        self.metadata[3].t_prime = 0.1
        self.metadata[3].observations = [0] * 2

        # Item 4: No t_prime, should be considered a "suspected loser"
        self.metadata[4].t_prime = None

        victims = policy.select_victims(self.metadata)

        # Expected victims:
        # 1. Item 4 (infinite distance because t_prime is None)
        # 2. Item 1 (Suspected Loser, has a larger distance than the Proven Loser)
        self.assertEqual(len(victims), self.num_to_evict)
        self.assertEqual(sorted(victims), [1, 4])

    def test_scu_fallback_eviction(self):
        """Test the SCU fallback to LRU when no t_prime is available."""
        scu_policy = SCUEvictionPolicy(self.max_size, 0.9, self.eviction_percentage)
        lru_policy = LRUEvictionPolicy(self.max_size, 0.9, self.eviction_percentage)

        # Ensure all t_prime values are None
        for meta in self.metadata:
            meta.t_prime = None

        # Update last_accessed to create a clear LRU order
        # Items 2 and 3 will be the least recently used
        lru_policy.update_eviction_metadata(self.metadata[0])
        time.sleep(0.01)
        lru_policy.update_eviction_metadata(self.metadata[1])
        time.sleep(0.01)
        lru_policy.update_eviction_metadata(self.metadata[4])

        victims = scu_policy.select_victims(self.metadata)

        # Fallback should use LRU, evicting the least recently used items
        self.assertEqual(len(victims), self.num_to_evict)
        self.assertEqual(sorted(victims), [2, 3])

    def test_cost_aware_matches_lru_with_no_cost_data(self):
        """With no cost data, CostAwareEvictionPolicy should behave like LRU."""
        policy = CostAwareEvictionPolicy(self.max_size, 0.9, self.eviction_percentage)

        # Simulate access to items 0 and 1, making them the most recently used
        policy.update_eviction_metadata(self.metadata[0])
        time.sleep(0.01)
        policy.update_eviction_metadata(self.metadata[1])

        victims = policy.select_victims(self.metadata)

        # Same result as plain LRU: items 2 and 3 are the least recently used
        self.assertEqual(len(victims), self.num_to_evict)
        self.assertEqual(sorted(victims), [2, 3])

    def test_cost_aware_protects_expensive_items(self):
        """CostAwareEvictionPolicy should protect stale but expensive items."""
        policy = CostAwareEvictionPolicy(
            self.max_size, 0.9, self.eviction_percentage, cost_weight=0.9
        )

        # Simulate access to items 0 and 1, making them the most recently used.
        # This leaves items 2, 3, and 4 as the stalest, in that order (item 2
        # is the most stale, matching the plain-LRU test which evicts [2, 3]).
        policy.update_eviction_metadata(self.metadata[0])
        time.sleep(0.01)
        policy.update_eviction_metadata(self.metadata[1])

        for meta in self.metadata:
            meta.cost = 0.0
        # Item 2 was very expensive to generate, so it should be protected
        # from eviction in favor of the next stalest (cheap) item, item 4.
        self.metadata[2].cost = 100.0

        victims = policy.select_victims(self.metadata)

        self.assertEqual(len(victims), self.num_to_evict)
        self.assertNotIn(2, victims)
        self.assertEqual(sorted(victims), [3, 4])

    def test_cost_aware_cost_weight_zero_equals_lru(self):
        """At cost_weight=0.0, CostAwareEvictionPolicy must reduce to plain
        LRU, ignoring cost entirely -- even when cost data would otherwise
        protect the stalest item."""
        policy = CostAwareEvictionPolicy(
            self.max_size, 0.9, self.eviction_percentage, cost_weight=0.0
        )

        policy.update_eviction_metadata(self.metadata[0])
        time.sleep(0.01)
        policy.update_eviction_metadata(self.metadata[1])

        # Item 2 is the stalest and would normally be evicted first under
        # LRU. Give it a huge cost -- at cost_weight=0.0 this must have no
        # effect on the outcome.
        for meta in self.metadata:
            meta.cost = 0.0
        self.metadata[2].cost = 1000.0

        victims = policy.select_victims(self.metadata)

        self.assertEqual(len(victims), self.num_to_evict)
        self.assertEqual(sorted(victims), [2, 3])

    def test_cost_aware_cost_weight_one_ignores_staleness(self):
        """At cost_weight=1.0, eviction should be driven purely by cost,
        even when that means evicting the freshest (least stale) items."""
        policy = CostAwareEvictionPolicy(
            self.max_size, 0.9, self.eviction_percentage, cost_weight=1.0
        )

        # Items are in creation order 0 (oldest) .. 4 (newest), so plain
        # LRU would evict 0 and 1. Make the newest items (3, 4) the
        # cheapest -- at cost_weight=1.0 they must be evicted first despite
        # being the freshest.
        self.metadata[0].cost = 1000.0
        self.metadata[1].cost = 1000.0
        self.metadata[2].cost = 50.0
        self.metadata[3].cost = 0.1
        self.metadata[4].cost = 0.0

        victims = policy.select_victims(self.metadata)

        self.assertEqual(len(victims), self.num_to_evict)
        self.assertEqual(sorted(victims), [3, 4])

    def test_cost_aware_invalid_cost_weight_defaults_to_half(self):
        """An out-of-range cost_weight should be clamped to the documented
        default of 0.5 rather than silently producing an invalid policy."""
        policy_too_low = CostAwareEvictionPolicy(
            self.max_size, 0.9, self.eviction_percentage, cost_weight=-0.5
        )
        policy_too_high = CostAwareEvictionPolicy(
            self.max_size, 0.9, self.eviction_percentage, cost_weight=1.5
        )

        self.assertEqual(policy_too_low.cost_weight, 0.5)
        self.assertEqual(policy_too_high.cost_weight, 0.5)

    def test_cost_aware_min_max_normalize(self):
        """_min_max_normalize should scale values to [0, 1], mapping the
        minimum to 0.0 and the maximum to 1.0, and collapse to all-zero
        when every value is equal (no variation to distinguish them by)."""
        normalized = CostAwareEvictionPolicy._min_max_normalize([10.0, 20.0, 30.0])
        self.assertEqual(normalized, [0.0, 0.5, 1.0])

        normalized_equal = CostAwareEvictionPolicy._min_max_normalize([5.0, 5.0, 5.0])
        self.assertEqual(normalized_equal, [0.0, 0.0, 0.0])

    def test_cost_aware_compute_priority(self):
        """_compute_priority should blend normalized staleness and
        (inverted) normalized cost by cost_weight, exactly per the
        documented formula."""
        policy = CostAwareEvictionPolicy(
            self.max_size, 0.9, self.eviction_percentage, cost_weight=0.25
        )
        # priority = (1 - 0.25) * staleness + 0.25 * (1 - cost)
        priority = policy._compute_priority(
            normalized_staleness=0.8, normalized_cost=0.4
        )
        self.assertAlmostEqual(priority, 0.75 * 0.8 + 0.25 * (1 - 0.4))

    def test_arc_evicts_stalest_when_all_items_unreferenced(self):
        """With no items ever re-referenced (hit_count 0 for all), every
        item lands in T1, so ARC should fall back to plain staleness
        ordering -- the two oldest items, matching the LRU/FIFO baseline."""
        policy = ARCEvictionPolicy(self.max_size, 0.9, self.eviction_percentage)
        for meta in self.metadata:
            meta.observations = []

        victims = policy.select_victims(self.metadata)

        self.assertEqual(len(victims), self.num_to_evict)
        self.assertEqual(sorted(victims), [0, 1])

    def test_arc_draws_victims_from_both_t1_and_t2(self):
        """Items referenced at least once (hit_count >= 1) go to T2 and are
        ordered by (hit_count, staleness) there; unreferenced items stay in
        T1 ordered by staleness. Victims should be drawn from both lists
        according to the adaptive split `p`, not just the stalest overall."""
        policy = ARCEvictionPolicy(
            self.max_size, 0.9, self.eviction_percentage, initial_p=0.5, p_step=0.1
        )
        # Items 0 and 1 have been referenced once (2 seed + 1 real
        # observation) -> T2. Items 2, 3, 4 are unreferenced -> T1.
        self.metadata[0].observations = [0, 0, 0]
        self.metadata[1].observations = [0, 0, 0]
        self.metadata[2].observations = []
        self.metadata[3].observations = []
        self.metadata[4].observations = []

        victims = policy.select_victims(self.metadata)

        # p starts at 0.5 and drops to 0.4 on this first call (no promotion
        # signal yet), giving 1 victim from T1 (stalest: item 2) and 1 from
        # T2 (stalest within T2: item 0, since both have equal hit_count).
        self.assertEqual(len(victims), self.num_to_evict)
        self.assertEqual(sorted(victims), [0, 2])

    def test_arc_hit_count_subtracts_seed_observations(self):
        """_hit_count should discount the 2 synthetic seed observations
        every item starts with, and never go negative."""
        meta = MagicMock()

        meta.observations = [0, 0]
        self.assertEqual(ARCEvictionPolicy._hit_count(meta), 0)

        meta.observations = [0, 0, 0, 0, 0]
        self.assertEqual(ARCEvictionPolicy._hit_count(meta), 3)

        meta.observations = [0]
        self.assertEqual(ARCEvictionPolicy._hit_count(meta), 0)

    def test_arc_adapts_p_up_on_promotion(self):
        """A T1 item from the previous round reappearing in T2 (promoted)
        should push `p` up by `p_step`, mirroring a B1 ghost hit in
        textbook ARC."""
        policy = ARCEvictionPolicy(
            self.max_size, 0.9, self.eviction_percentage, initial_p=0.5, p_step=0.1
        )
        policy._prev_t1_ids = {0, 1}

        policy._adapt_p(t1=[], t2=[self.metadata[0]])

        self.assertAlmostEqual(policy.p, 0.6)

    def test_arc_adapts_p_down_without_promotion(self):
        """No overlap between the previous round's T1 and the current T2
        (no promotions) should push `p` down by `p_step`, mirroring a B2
        ghost hit in textbook ARC."""
        policy = ARCEvictionPolicy(
            self.max_size, 0.9, self.eviction_percentage, initial_p=0.5, p_step=0.1
        )
        policy._prev_t1_ids = {0, 1}

        policy._adapt_p(t1=[self.metadata[2]], t2=[self.metadata[3]])

        self.assertAlmostEqual(policy.p, 0.4)

    def test_arc_p_clamped_to_valid_range(self):
        """`p` should never leave [0, 1] regardless of how many consecutive
        promotion/no-promotion rounds occur."""
        policy = ARCEvictionPolicy(
            self.max_size, 0.9, self.eviction_percentage, initial_p=0.95, p_step=0.5
        )
        policy._prev_t1_ids = {0}
        policy._adapt_p(t1=[], t2=[self.metadata[0]])  # promotion: p -> 1.0, clamped
        self.assertEqual(policy.p, 1.0)

        policy._prev_t1_ids = {0}
        policy._adapt_p(t1=[self.metadata[1]], t2=[])  # no promotion: p -> 0.5
        policy._adapt_p(t1=[self.metadata[1]], t2=[])  # no promotion: p -> 0.0, clamped
        self.assertEqual(policy.p, 0.0)

    def test_arc_invalid_params_default(self):
        """Out-of-range initial_p or p_step should clamp to the documented
        defaults (0.5 and 0.1) rather than producing an invalid policy."""
        policy_bad_p = ARCEvictionPolicy(
            self.max_size, 0.9, self.eviction_percentage, initial_p=-0.5
        )
        policy_bad_step = ARCEvictionPolicy(
            self.max_size, 0.9, self.eviction_percentage, p_step=1.5
        )

        self.assertEqual(policy_bad_p.p, 0.5)
        self.assertEqual(policy_bad_step.p_step, 0.1)

    def test_cost_aware_handles_negative_and_zero_cost(self):
        """Negative or zero costs (e.g. from a bad cost measurement) should
        not crash normalization or eviction -- they're unusual but valid
        inputs."""
        policy = CostAwareEvictionPolicy(
            self.max_size, 0.9, self.eviction_percentage, cost_weight=0.9
        )
        self.metadata[0].cost = -5.0
        self.metadata[1].cost = 0.0
        self.metadata[2].cost = 0.0
        self.metadata[3].cost = 0.0
        self.metadata[4].cost = 0.0

        victims = policy.select_victims(self.metadata)

        self.assertEqual(len(victims), self.num_to_evict)

if __name__ == "__main__":
    unittest.main()
