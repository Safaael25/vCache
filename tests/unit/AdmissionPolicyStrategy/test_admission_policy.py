import unittest
from unittest.mock import MagicMock

from vcache.vcache_core.cache.admission_policy.strategies.always_admit import (
    AlwaysAdmitPolicy,
)
from vcache.vcache_core.cache.admission_policy.strategies.similarity_lfu import (
    SimilarityLFUAdmissionPolicy,
)
from vcache.vcache_core.cache.cache import Cache


class TestAlwaysAdmitPolicy(unittest.TestCase):
    def test_always_admits(self):
        """AlwaysAdmitPolicy should admit every embedding, unconditionally."""
        policy = AlwaysAdmitPolicy()
        self.assertTrue(policy.should_admit([1.0, 0.0, 0.0]))
        self.assertTrue(policy.should_admit([0.0, 0.0, 0.0]))


class TestSimilarityLFUAdmissionPolicy(unittest.TestCase):
    def test_first_sighting_is_not_admitted(self):
        """A never-seen-before embedding should be put on probation, not admitted."""
        policy = SimilarityLFUAdmissionPolicy(admission_similarity_threshold=0.9)
        self.assertFalse(policy.should_admit([1.0, 0.0, 0.0]))

    def test_similar_second_sighting_is_admitted(self):
        """A repeat of a probated embedding (above the similarity threshold) is admitted."""
        policy = SimilarityLFUAdmissionPolicy(admission_similarity_threshold=0.9)
        policy.should_admit([1.0, 0.0, 0.0])
        self.assertTrue(policy.should_admit([1.0, 0.0001, 0.0]))

    def test_dissimilar_second_sighting_is_not_admitted(self):
        """An unrelated embedding should not be promoted by an unrelated probation entry."""
        policy = SimilarityLFUAdmissionPolicy(admission_similarity_threshold=0.9)
        policy.should_admit([1.0, 0.0, 0.0])
        self.assertFalse(policy.should_admit([0.0, 1.0, 0.0]))

    def test_promoted_entry_is_removed_from_probation(self):
        """Once promoted, an entry should not still be sitting on probation."""
        policy = SimilarityLFUAdmissionPolicy(admission_similarity_threshold=0.9)
        policy.should_admit([1.0, 0.0, 0.0])
        policy.should_admit([1.0, 0.0001, 0.0])  # promotes and removes the entry
        self.assertEqual(len(policy._probation), 0)

    def test_probation_capacity_evicts_oldest(self):
        """When probation is full, the oldest entry is dropped to make room."""
        policy = SimilarityLFUAdmissionPolicy(
            admission_similarity_threshold=0.9, probation_capacity=2
        )
        policy.should_admit([1.0, 0.0, 0.0])
        policy.should_admit([0.0, 1.0, 0.0])
        policy.should_admit([0.0, 0.0, 1.0])  # probation was full -> oldest dropped

        self.assertEqual(len(policy._probation), 2)
        # The first entry should have been evicted, so its repeat is not admitted.
        self.assertFalse(policy.should_admit([1.0, 0.0, 0.0]))

    def test_zero_vector_is_never_admitted(self):
        """A zero-norm embedding can't be compared by cosine similarity, so it's rejected."""
        policy = SimilarityLFUAdmissionPolicy()
        self.assertFalse(policy.should_admit([0.0, 0.0, 0.0]))


class TestCacheAdmissionPolicyWiring(unittest.TestCase):
    def setUp(self):
        self.embedding_store = MagicMock()
        self.embedding_store.add_embedding.return_value = 42
        self.embedding_engine = MagicMock()
        self.embedding_engine.get_embedding.return_value = [1.0, 0.0, 0.0]
        self.eviction_policy = MagicMock()

    def test_defaults_to_always_admit(self):
        """With no admission_policy given, Cache.add() should behave as before (always admit)."""
        cache = Cache(
            embedding_store=self.embedding_store,
            embedding_engine=self.embedding_engine,
            eviction_policy=self.eviction_policy,
        )
        result = cache.add(prompt="hi", response="hello", id_set=1)
        self.assertEqual(result, 42)
        self.embedding_store.add_embedding.assert_called_once()

    def test_rejecting_admission_policy_skips_insertion(self):
        """When the admission policy declines, Cache.add() should not insert anything."""
        rejecting_policy = MagicMock()
        rejecting_policy.should_admit.return_value = False
        cache = Cache(
            embedding_store=self.embedding_store,
            embedding_engine=self.embedding_engine,
            eviction_policy=self.eviction_policy,
            admission_policy=rejecting_policy,
        )
        result = cache.add(prompt="hi", response="hello", id_set=1)
        self.assertEqual(result, -1)
        self.embedding_store.add_embedding.assert_not_called()

    def test_accepting_admission_policy_allows_insertion(self):
        """When the admission policy admits, Cache.add() should behave normally."""
        accepting_policy = MagicMock()
        accepting_policy.should_admit.return_value = True
        cache = Cache(
            embedding_store=self.embedding_store,
            embedding_engine=self.embedding_engine,
            eviction_policy=self.eviction_policy,
            admission_policy=accepting_policy,
        )
        result = cache.add(prompt="hi", response="hello", id_set=1)
        self.assertEqual(result, 42)
        self.embedding_store.add_embedding.assert_called_once()


if __name__ == "__main__":
    unittest.main()
