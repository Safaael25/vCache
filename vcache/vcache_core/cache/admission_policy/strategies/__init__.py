from vcache.vcache_core.cache.admission_policy.strategies.always_admit import (
    AlwaysAdmitPolicy,
)
from vcache.vcache_core.cache.admission_policy.strategies.similarity_lfu import (
    SimilarityLFUAdmissionPolicy,
)

__all__ = [
    "AlwaysAdmitPolicy",
    "SimilarityLFUAdmissionPolicy",
]
