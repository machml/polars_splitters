from typing import Callable, Optional

import polars as pl
import pytest

from tests.utils.data_generators import (
    get_uniformly_distributed_combinations_of_two_bools,
)


@pytest.fixture
def df_ubools():
    """Build a dataframe with two boolean columns that are both marginally and jointly uniformly distributed."""

    def _df(from_lazy: bool = False, n: int = 400, seed: Optional[int] = 273):
        treatment_outcome_pairs = get_uniformly_distributed_combinations_of_two_bools(n=n, seed=seed)
        df_lazy = pl.LazyFrame(
            {
                "row_num_as_float": [float(row_num) for row_num in range(n)],
                # treatment and outcome are each 50% 0s and 50% 1s; their combinations are uniformly distributed
                "treatment": [pair_values[0] for pair_values in treatment_outcome_pairs],
                "outcome": [pair_values[1] for pair_values in treatment_outcome_pairs],
            }
        ).with_columns(
            [
                (pl.col("row_num_as_float") * 10.0).alias("feature_1"),
                (pl.col("row_num_as_float") * 100.0).alias("feature_2"),
            ]
        )

        if from_lazy:
            return df_lazy
        else:
            return df_lazy.collect()

    return _df


@pytest.fixture
def df_basic() -> Callable[[bool], pl.DataFrame | pl.LazyFrame]:
    """Define a dataframe with a single float feature, and integer-typed treatment and outcome columns,
    so that treatment and outcome combined present 3 strata with minimum size 3,
    and treatment and outcome individually present 2 strata with minimum size 3.
    """

    def _df(from_lazy: bool = False, n_cols: int = 2) -> pl.DataFrame | pl.LazyFrame:
        df = pl.DataFrame(
            {
                "row_cnt_as_float": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "treatment": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                "outcome": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            }
        )

        if from_lazy:
            return df.lazy()
        else:
            return df

    return _df
