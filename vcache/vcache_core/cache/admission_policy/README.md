# Cache Admission Policies

An `EvictionPolicy` decides what gets removed once the cache is full. An `AdmissionPolicy` answers a different question: is a cache-miss item even worth caching in the first place? By default, `vCache` admits every miss (`AlwaysAdmitPolicy`), so plugging in an `AdmissionPolicy` is entirely opt-in and does not change existing behavior unless explicitly configured via `VCacheConfig(admission_policy=...)`.

## Always Admit

`AlwaysAdmitPolicy` is the default: every cache-miss item is cached. This preserves vCache's original behavior.

## Similarity-LFU

`SimilarityLFUAdmissionPolicy` is adapted from the idea behind TinyLFU (used e.g. by the Caffeine cache library): don't admit something until it has shown some sign of recurring. TinyLFU itself checks recurrence with a hashed frequency sketch over *exact* keys, which doesn't transfer to vCache directly -- two semantically identical queries with different wording hash completely differently, so an exact-key sketch can't see the relationship vCache's whole caching strategy is built around. This policy checks recurrence by semantic similarity instead: a cache-miss item is *not* admitted the first time it's seen. Its embedding is kept on a small, bounded probation list. If a similar embedding is seen again while the original is still on probation, the item is promoted (admitted) on that second sighting and removed from probation. If probation is full when a new, non-matching item arrives, the oldest entry is dropped to make room (a FIFO doorkeeper, in TinyLFU's terms).

The probation list is separate from the main cache -- it never touches the eviction policy or the vector index, so it doesn't consume any of the main cache's capacity.

### Empirical Results

Tested against a real 1591-row workload (real LmArena duplicate-query clusters, 80% ordinary / 10% expensive / 10% unique-noise mix) across all 6 eviction policies (LRU, FIFO, SCU, CostAware, CostAwareSCU, GPCA) and 7 MB-calibrated cache sizes, `SimilarityLFUAdmissionPolicy` shows a sharp, consistent split by cache pressure:

- **Tight caches (20-40MB, real eviction churn):** admission control helps every eviction policy substantially. E.g. at 40MB, average hit ratio across all 6 policies goes from 2.13% (AlwaysAdmit) to 5.82% (SimilarityLFU), and average precision (`expected_hit_ratio`) from 76.3% to 82.2%.
- **Generous caches (80-200MB, little eviction pressure):** it consistently *hurts* every policy instead. E.g. at 100MB, average hit ratio drops from 13.26% to 8.10%, precision from 97.7% to 87.0%.

This is expected from the mechanism itself: `SimilarityLFUAdmissionPolicy` never admits an item on its first sighting, so under real eviction pressure it stops one-off noise from ever burning a cache slot, letting genuine repeats survive eviction. But once the cache is large enough to hold most of the working set anyway, that same one-sighting delay just costs free hits, since there was no eviction pressure for it to protect against in the first place.

**Practical takeaway:** `SimilarityLFUAdmissionPolicy` is not a universal improvement -- it is a tight-cache-pressure tool. It should be enabled when the cache is small relative to the working set, and left as `AlwaysAdmitPolicy` (the default) when the cache is generously sized.
