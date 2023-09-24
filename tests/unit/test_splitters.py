from collections import Counter

import polars as pl
import pytest
from pytest_check import check

from src.random_split.core.splitters import split_into_subsets, validate_rel_sizes

SEED = 273
df_pl = pl.DataFrame | pl.LazyFrame


def test_validate_rel_sizes():
    def mock_splitter(x, rel_sizes):
        return None

    # Test valid input
    rel_sizes = (0.7, 0.2, 0.1)
    check.is_none(validate_rel_sizes(mock_splitter)(None, rel_sizes=rel_sizes))

    # Test valid input
    rel_sizes = {"train": 0.7, "val": 0.2, "test": 0.1}
    check.is_none(validate_rel_sizes(mock_splitter)(None, rel_sizes=rel_sizes))

    # Test invalid input: not summing to one
    rel_sizes = (0.7, 0.2)
    with check.raises(ValueError):
        validate_rel_sizes(mock_splitter)(None, rel_sizes=rel_sizes)

    # Test invalid input: fraction larger than 1
    rel_sizes = (0.7, 0.2, 0.8)
    with check.raises(ValueError):
        validate_rel_sizes(mock_splitter)(None, rel_sizes=rel_sizes)

    # Test invalid input: tuple with negative value
    rel_sizes = (0.7, 0.8, -0.5)
    with check.raises(ValueError):
        validate_rel_sizes(mock_splitter)(None, rel_sizes=rel_sizes)

    # Test invalid input: tuple with non-float
    rel_sizes = (0.2, 0.3, "0.5")
    with check.raises(TypeError):
        validate_rel_sizes(mock_splitter)(None, rel_sizes=rel_sizes)


class TestSplitIntoSubsets:
    @pytest.mark.parametrize("from_lazy", [False, True])
    @pytest.mark.parametrize("rel_sizes", [(0.5, 0.1, 0.1), {"train": 0.5, "val": 0.1, "test": 0.1}])
    def test_handling_rel_sizes_not_summing_to_one(self, df_basic, from_lazy, rel_sizes):
        with check.raises(ValueError):
            split_into_subsets(df=df_basic(from_lazy), rel_sizes=rel_sizes)

    @pytest.mark.parametrize("from_lazy", [False, True])
    @pytest.mark.parametrize("rel_sizes", [(0.7, 0.2, 0.1), {"train": 0.7, "val": 0.2, "test": 0.1}])
    def test_as_tuples(self, df_basic, from_lazy, rel_sizes):
        subsets = split_into_subsets(df_basic(from_lazy), rel_sizes=rel_sizes)
        check.equal(len(subsets), 3)
        check.equal(len(subsets[0]), 7)
        check.equal(len(subsets[1]), 2)
        check.equal(len(subsets[2]), 1)

    @pytest.mark.parametrize("from_lazy", [False, True])
    @pytest.mark.parametrize("shuffle", [False, True])
    @pytest.mark.parametrize("as_dict", [False, True])
    @pytest.mark.parametrize("rel_sizes", [(0.7, 0.2, 0.1), {"train": 0.7, "val": 0.2, "test": 0.1}])
    def test_to_df(self, df_basic, from_lazy, rel_sizes, shuffle, as_dict):
        subsets = split_into_subsets(df=df_basic(from_lazy), rel_sizes=rel_sizes, shuffle=shuffle, as_dict=as_dict)

        if isinstance(rel_sizes, dict) and as_dict:
            check.equal(len(subsets), 3)
            check.equal(len(subsets["train"]), 7)
            check.equal(len(subsets["val"]), 2)
            check.equal(len(subsets["test"]), 1)
        else:
            check.equal(len(subsets[0]), 7)
            check.equal(len(subsets[1]), 2)
            check.equal(len(subsets[2]), 1)

    @pytest.mark.parametrize("from_lazy", [False, True])
    @pytest.mark.parametrize("shuffle", [False, True])
    @pytest.mark.parametrize("as_dict", [False, True])
    @pytest.mark.parametrize("rel_sizes", [(0.7, 0.2, 0.1), {"train": 0.7, "val": 0.2, "test": 0.1}])
    def test_to_lazy_df(self, df_basic, from_lazy, rel_sizes, shuffle, as_dict):
        subsets = split_into_subsets(
            df=df_basic(from_lazy), rel_sizes=rel_sizes, shuffle=shuffle, as_dict=as_dict, as_lazy=True
        )

        if isinstance(rel_sizes, dict) and as_dict:
            for subset in subsets.values():
                check.is_instance(subset, pl.LazyFrame)
            check.equal(len(subsets), 3)
            check.equal(len(subsets["train"].collect()), 7)
            check.equal(len(subsets["val"].collect()), 2)
            check.equal(len(subsets["test"].collect()), 1)
        else:
            for subset in subsets:
                check.is_instance(subset, pl.LazyFrame)
            check.equal(len(subsets[0].collect()), 7)
            check.equal(len(subsets[1].collect()), 2)
            check.equal(len(subsets[2].collect()), 1)

    @pytest.mark.parametrize("from_lazy", [False, True])
    @pytest.mark.parametrize("shuffle", [False, True])
    @pytest.mark.parametrize("rel_sizes", [(0.7, 0.2, 0.1), {"train": 0.7, "val": 0.2, "test": 0.1}])
    def test_shuffle(self, df_basic, from_lazy, rel_sizes, shuffle):
        subsets = split_into_subsets(df=df_basic(from_lazy), rel_sizes=rel_sizes, shuffle=shuffle)

        df_ = df_basic(from_lazy)
        if from_lazy:
            df_ = df_.collect()

        if shuffle:
            check.is_true(~subsets[0].frame_equal(df_[0:7]))
            check.is_true(~subsets[1].frame_equal(df_[7:9]))
            check.is_true(~subsets[2].frame_equal(df_[9]))
        else:
            check.is_true(subsets[0].frame_equal(df_[0:7]))
            check.is_true(subsets[1].frame_equal(df_[7:9]))
            check.is_true(subsets[2].frame_equal(df_[9]))

    @pytest.mark.parametrize("from_lazy", [False, True])
    @pytest.mark.parametrize(
        "stratify_by, strata_counts_train, strata_counts_val, strata_counts_test",
        [
            (
                "treatment",
                {0: 140, 1: 140},
                {0: 40, 1: 40},
                {0: 20, 1: 20},
            ),
            (
                ["treatment", "outcome"],
                {(0, 1): 70, (1, 0): 70, (1, 1): 70, (0, 0): 70},
                {(0, 1): 20, (1, 0): 20, (1, 1): 20, (0, 0): 20},
                {(0, 1): 10, (1, 0): 10, (1, 1): 10, (0, 0): 10},
            ),
        ],
    )
    def test_stratification(
        self, df_ubools, from_lazy, stratify_by, strata_counts_train, strata_counts_val, strata_counts_test
    ):
        subsets = split_into_subsets(
            df=df_ubools(from_lazy=from_lazy, n=400),
            rel_sizes={"train": 0.7, "val": 0.2, "test": 0.1},
            stratify_by=stratify_by,
            as_dict=True,
            shuffle=False,
        )

        df_train = subsets["train"]
        df_val = subsets["val"]
        df_test = subsets["test"]

        if stratify_by == "treatment":
            unique_combinations_count_train = Counter(df_train["treatment"])
            unique_combinations_count_val = Counter(df_val["treatment"])
            unique_combinations_count_test = Counter(df_test["treatment"])
        elif stratify_by == ["treatment", "outcome"]:
            unique_combinations_count_train = Counter(zip(df_train["treatment"], df_train["outcome"]))
            unique_combinations_count_val = Counter(zip(df_val["treatment"], df_val["outcome"]))
            unique_combinations_count_test = Counter(zip(df_test["treatment"], df_test["outcome"]))
        else:
            pass

        if stratify_by is not None:
            check.equal(unique_combinations_count_train, Counter(strata_counts_train))
            check.equal(unique_combinations_count_val, Counter(strata_counts_val))
            check.equal(unique_combinations_count_test, Counter(strata_counts_test))
        else:
            pass
