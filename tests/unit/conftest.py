from typing import Callable

import polars as pl
import pytest

from tests.utils.data_generators import (
    get_uniformly_distributed_combinations_of_two_bools,
)


@pytest.fixture
def df_ubools():
    """Build a dataframe with two boolean columns that are both marginally and jointly uniformly distributed."""

    def _df_ubools(from_lazy: bool = False, n: int = 400):
        treatment_outcome_pairs = get_uniformly_distributed_combinations_of_two_bools(n=n)
        df_lazy = pl.LazyFrame(
            {
                "row": [row_num for row_num in range(n)],
                # treatment and outcome are each 50% 0s and 50% 1s; their combinations are uniformly distributed
                "treatment": [pair_values[0] for pair_values in treatment_outcome_pairs],
                "outcome": [pair_values[1] for pair_values in treatment_outcome_pairs],
            }
        ).with_columns([(pl.col("row") * 10.0).alias("feature_1"), (pl.col("row") * 100.0).alias("feature_2")])

        if from_lazy:
            return df_lazy
        else:
            return df_lazy.collect()

    return _df_ubools


@pytest.fixture
def df_basic() -> Callable[[bool], pl.DataFrame | pl.LazyFrame]:
    def _df(from_lazy: bool = False, n_cols: int = 2) -> pl.DataFrame | pl.LazyFrame:
        df_lazy = pl.LazyFrame({f"col_{i}": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] for i in range(n_cols)})
        if from_lazy:
            return df_lazy
        else:
            return df_lazy.collect()

    return _df
