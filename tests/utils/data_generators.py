import random
from itertools import product as cartesian_product
from typing import List, Optional, Tuple


def get_uniformly_distributed_combinations_of_two_bools(
    n: int, k: int = 2, seed: Optional[int] = 273
) -> Tuple[List[int], List[int]]:
    """
    Get all possible combinations of k booleans.

    Attention: n must be a multiple of 2 ** k.
    """
    n_combs = 2**k

    if n % n_combs != 0:
        raise ValueError(f"n must be a multiple of {n_combs}, but got {n}")

    combs = list(cartesian_product(list(range(k)), repeat=2))
    combs_repeated = combs * (n // len(combs))

    if seed is not None:
        random.seed(seed)
    combs_uniformly_distributed = random.sample(combs_repeated, k=len(combs_repeated))

    return combs_uniformly_distributed
