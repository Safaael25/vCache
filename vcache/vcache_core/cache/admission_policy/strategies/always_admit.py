from typing import List

from vcache.vcache_core.cache.admission_policy.admission_policy import AdmissionPolicy


class AlwaysAdmitPolicy(AdmissionPolicy):
    """
    The default admission policy: every cache-miss item is admitted.

    This preserves vCache's original behavior (no admission filtering at
    all), so plugging in an `AdmissionPolicy` is fully opt-in.
    """

    def should_admit(self, embedding: List[float]) -> bool:
        """Always admits the item.

        Args:
            embedding: The embedding vector of the prompt that just missed
                (unused).

        Returns:
            True, unconditionally.
        """
        return True
