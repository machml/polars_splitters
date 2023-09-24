import random
from collections import Counter
from itertools import combinations
from itertools import product as cartesian_product
from typing import List, Tuple

import polars as pl
import pytest
from pytest_check import check

from src.random_split.core.splitters import get_k_folds

SEED = 3006


def get_uniformly_distributed_combinations_of_two_bools(n: int, k: int = 2) -> Tuple[List[int], List[int]]:
    """
    Get all possible combinations of k booleans.

    Attention: n must be a multiple of 2 ** k.
    """
    n_combs = 2**k

    if n % n_combs != 0:
        raise ValueError(f"n must be a multiple of {n_combs}, but got {n}")

    combs = list(cartesian_product(list(range(k)), repeat=2))
    combs_repeated = combs * (n // len(combs))
    combs_uniformly_distributed = random.sample(combs_repeated, k=len(combs_repeated))

    return combs_uniformly_distributed


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


class TestGetKFolds:
    @pytest.fixture
    def df_lazy(self) -> pl.LazyFrame:
        n = 400

        treatment_outcome_pairs = get_uniformly_distributed_combinations_of_two_bools(n=n)
        return pl.LazyFrame(
            {
                "row": [row_num for row_num in range(n)],
                # treatment and outcome are each 50% 0s and 50% 1s; their combinations are uniformly distributed
                "treatment": [pair_values[0] for pair_values in treatment_outcome_pairs],
                "outcome": [pair_values[1] for pair_values in treatment_outcome_pairs],
            }
        )
        if from_lazy:
            return df_lazy
        else:
            return df_lazy.collect()

    return _df


class TestSplitIntoSubsets:
    @pytest.mark.parametrize("from_lazy", [False, True])
    @pytest.mark.parametrize("rel_sizes", [(0.5, 0.1, 0.1), {"train": 0.5, "val": 0.1, "test": 0.1}])
    def test_handling_rel_sizes_not_summing_to_one(self, df, from_lazy, rel_sizes):
        with check.raises(ValueError):
            split_into_subsets(df=df(from_lazy), rel_sizes=rel_sizes)

    @pytest.mark.parametrize("from_lazy", [False, True])
    @pytest.mark.parametrize("rel_sizes", [(0.7, 0.2, 0.1), {"train": 0.7, "val": 0.2, "test": 0.1}])
    def test_case_df_to_dfs(self, df, from_lazy, rel_sizes):
        subsets = split_into_subsets(df(from_lazy), rel_sizes=rel_sizes)
        check.equal(len(subsets), 3)
        check.equal(len(subsets[0]), 7)
        check.equal(len(subsets[1]), 2)
        check.equal(len(subsets[2]), 1)

    @pytest.fixture
    def df(self, df_lazy) -> pl.DataFrame:
        return df_lazy.collect()

    @pytest.fixture
    def folds(self, df, k, stratify_by, lazy, shuffle):
        return get_k_folds(df, k=k, stratify_by=stratify_by, lazy=lazy, shuffle=shuffle, seed=SEED)

    @pytest.mark.parametrize(
        "k, stratify_by, lazy, shuffle",
        [(2, "treatment", True, False), (3, ["treatment", "outcome"], False, True), (3, None, True, True)],
    )
    def test_k_folds(self, df, k, stratify_by, lazy, shuffle, folds):
        assert len(folds) == k
        assert set(folds.keys()) == set(range(k))

    @pytest.mark.parametrize(
        "k, stratify_by, lazy, shuffle",
        [(2, "treatment", True, False), (3, ["treatment", "outcome"], False, True), (3, None, True, True)],
    )
    def test_k_folds_content_type(self, df, k, stratify_by, lazy, shuffle, folds):
        if lazy:
            expected_type = pl.LazyFrame
        else:
            expected_type = pl.DataFrame

        for i in range(k):
            check.is_true(isinstance(folds[i]["train"], expected_type))
            check.is_true(isinstance(folds[i]["eval"], expected_type))

    @pytest.mark.parametrize(
        "k, stratify_by, lazy, shuffle",
        [(2, "treatment", True, False), (3, ["treatment", "outcome"], True, True), (3, None, True, True)],
    )
    def test_no_leakage(self, df, k, stratify_by, lazy, shuffle, folds):
        if lazy:
            folds = {
                k: {"train": folds[k]["train"].collect(), "eval": folds[k]["eval"].collect()} for k in folds.keys()
            }

        for i, j in combinations(range(k), 2):
            check.equal(set(folds[i]["eval"]["row"]).intersection(set(folds[j]["eval"]["row"])), set())
            check.equal(set(folds[i]["eval"]["row"]).intersection(set(folds[i]["train"]["row"])), set())

    @pytest.mark.parametrize(
        "k, stratify_by, lazy, shuffle, strata_counts_train, strata_counts_eval",
        [
            (
                2,
                "treatment",
                True,
                False,
                {0: 100, 1: 100},
                {0: 100, 1: 100},
            ),
            (
                3,
                ["treatment", "outcome"],
                True,
                True,
                {(0, 1): 67, (1, 0): 67, (1, 1): 67, (0, 0): 67},
                {(0, 1): 33, (1, 0): 33, (1, 1): 33, (0, 0): 33},
            ),
            (
                3,
                "treatment",
                True,
                True,
                {0: 134, 1: 134},
                {0: 66, 1: 66},
            ),
            # (3, None, True, True, None, None),
        ],
    )
    def test_stratification(self, df, k, stratify_by, lazy, shuffle, strata_counts_train, strata_counts_eval, folds):
        if lazy:
            folds = {
                k: {"train": folds[k]["train"].collect(), "eval": folds[k]["eval"].collect()} for k in folds.keys()
            }

        for i in range(k):
            fold_train = folds[i]["train"]
            fold_eval = folds[i]["eval"]

            if stratify_by == "treatment":
                unique_combinations_count_train = Counter(fold_train["treatment"])
                unique_combinations_count_eval = Counter(fold_eval["treatment"])
            elif stratify_by == ["treatment", "outcome"]:
                unique_combinations_count_train = Counter(zip(fold_train["treatment"], fold_train["outcome"]))
                unique_combinations_count_eval = Counter(zip(fold_eval["treatment"], fold_eval["outcome"]))
            else:
                pass

            if stratify_by is not None:
                check.equal(unique_combinations_count_train, Counter(strata_counts_train))
                check.equal(unique_combinations_count_eval, Counter(strata_counts_eval))
            else:
                pass
