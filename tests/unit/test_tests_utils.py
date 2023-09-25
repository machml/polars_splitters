from collections import Counter

import pytest
from pytest_check import check

from tests.utils.data_generators import get_uniformly_distributed_combinations_of_two_bools


@pytest.mark.parametrize("k", [2])
@pytest.mark.parametrize("n", [100, 400, int(1e4), 3333])
def test_get_uniformly_distributed_combinations_of_two_bools(n, k):
    if n % (2**k) == 0:
        treatment_outcome_pairs = get_uniformly_distributed_combinations_of_two_bools(n=n, k=k)

        treatments = [pair_values[0] for pair_values in treatment_outcome_pairs]
        outcomes = [pair_values[1] for pair_values in treatment_outcome_pairs]

        check.equal(Counter(treatments), {0: n // 2, 1: n // 2})
        check.equal(Counter(outcomes), {0: n // 2, 1: n // 2})
        check.equal(
            Counter(zip(treatments, outcomes)), {(0, 0): n // 4, (1, 1): n // 4, (0, 1): n // 4, (1, 0): n // 4}
        )
