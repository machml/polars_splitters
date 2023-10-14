from typing import Dict, List, Optional, Tuple, Union

from loguru import logger
from polars import DataFrame, Int64, LazyFrame, count, int_range

from polars_splitters.utils.decorators import (
    type_enforcer_train_eval,
    validate_splitting_train_eval,
)

df_pl = DataFrame | LazyFrame

__all__ = [
    "split_into_train_eval",
    "split_into_k_folds",
]


@logger.catch
@type_enforcer_train_eval
@validate_splitting_train_eval
def split_into_train_eval(
    df: LazyFrame | DataFrame,
    eval_rel_size: float,
    stratify_by: Optional[str | List[str]] = None,
    shuffle: Optional[bool] = True,
    seed: Optional[int] = 273,
    as_lazy: Optional[bool] = False,
    validate: Optional[bool] = True,
    rel_size_deviation_tolerance: Optional[float] = 0.1,
) -> Tuple[LazyFrame, LazyFrame] | Tuple[DataFrame, DataFrame]:
    r"""
    Split a dataset into non-overlapping train and eval sets, optionally stratifying by a column or list of columns.
    Thanks to its decorators, it includes  logging and some guardrails: type coercion as well as validation for the inputs and outputs.

    Parameters
    ----------
    df : LazyFrame | DataFrame
        The polars DataFrame to split.
    eval_rel_size : float
        The targeted relative size of the eval set. Must be between 0.0 and 1.0.
    stratify_by : str | List[str], optional. Defaults to None.
        The column names to use for stratification.
        If None (default), stratification is not performed. Note: Stratification by float columns is not currently supported.
    shuffle : bool, optional. Defaults to True.
        Whether to shuffle the rows before splitting.
    seed : int, optional. Defaults to 273.
        The random seed to use in shuffling.
    as_lazy : bool, optional. Defaults to False.
        Whether to return the train and eval sets as LazyFrames (True) or DataFrames (False).
    validate : bool, optional. Defaults to True.
        Whether to validate the inputs and outputs.
    rel_size_deviation_tolerance : float, optional. Defaults to 0.1.
        Sets the maximum allowed abs(eval_rel_size_actual - eval_rel_size).
        When stratifying, the eval_rel_size_actual might deviate from eval_rel_size due to the fact that strata for the given data may not be perfectly divisible at the desired proportion (1-eval_rel_size, eval_rel_size).
        If validate is set to False, this parameter is ignored.

    Returns
    -------
    Tuple[LazyFrame, LazyFrame]
        _description_

    Raises
    ------
    NotImplementedError
        When trying to stratify by a float column.
    ValueError
        When the actual relative size of the eval set deviates from the requested relative size by more than the specified tolerance.
        Or when the size of the smallest set is smaller than the number of strata (unique row-wise combinations of values in the stratify_by columns).

    Examples
    --------
    >>> import polars as pl
    >>> from polars_splitters.core.splitters import split_into_train_eval
    >>> df = DataFrame(
    ...     {
    ...         "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    ...         "treatment": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    ...         "outcome":   [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    ...     }
    ... )
    >>> df_train, df_eval = split_into_train_eval(df, eval_rel_size=0.3, stratify_by=["treatment", "outcome"], shuffle=True, as_lazy=False)
    >>> print(df_train, df_eval, sep="\n\n")
    shape: (7, 3)
    ┌───────────┬───────────┬─────────┐
    │ feature_1 ┆ treatment ┆ outcome │
    │ ---       ┆ ---       ┆ ---     │
    │ f64       ┆ i64       ┆ i64     │
    ╞═══════════╪═══════════╪═════════╡
    │ 1.0       ┆ 0         ┆ 0       │
    │ 3.0       ┆ 0         ┆ 0       │
    │ 4.0       ┆ 0         ┆ 0       │
    │ 5.0       ┆ 0         ┆ 0       │
    │ 7.0       ┆ 1         ┆ 0       │
    │ 8.0       ┆ 1         ┆ 0       │
    │ 9.0       ┆ 1         ┆ 1       │
    └───────────┴───────────┴─────────┘

    shape: (3, 3)
    ┌───────────┬───────────┬─────────┐
    │ feature_1 ┆ treatment ┆ outcome │
    │ ---       ┆ ---       ┆ ---     │
    │ f64       ┆ i64       ┆ i64     │
    ╞═══════════╪═══════════╪═════════╡
    │ 2.0       ┆ 0         ┆ 0       │
    │ 6.0       ┆ 1         ┆ 0       │
    │ 10.0      ┆ 1         ┆ 1       │
    └───────────┴───────────┴─────────┘
    """
    idxs = int_range(0, count())
    if shuffle:
        idxs = idxs.shuffle(seed=seed)

    eval_size = (eval_rel_size * count()).round(0).clip_min(1).cast(Int64)

    if stratify_by:
        idxs = idxs.over(stratify_by)
        eval_size = eval_size.over(stratify_by)

    is_eval = idxs < eval_size

    df_train = df.filter(~is_eval)
    df_eval = df.filter(is_eval)

    return df_train, df_eval


def split_into_k_folds(
    df: LazyFrame | DataFrame,
    k: int,
    stratify_by: Union[str, List[str], Dict[str, float]] = None,
    shuffle: bool = True,
    as_lazy: bool = False,
    as_dict: bool = False,
    seed: int = 273,
) -> Tuple[df_pl, df_pl]:
    """Split a DataFrame or LazyFrame into k non-overlapping folds, allowing for stratification by a column or list of columns."""

    if k < 2:
        raise ValueError("k must be greater than 1")

    df = df.lazy()

    idxs = int_range(0, count())
    if shuffle:
        idxs = idxs.shuffle(seed=seed)

    eval_size = (count() / k).round(0).clip_min(1).cast(Int64)

    if stratify_by:
        idxs = idxs.over(stratify_by)
        eval_size = eval_size.over(stratify_by)

    folds = [{"train": None, "eval": None} for i in range(k)]
    for i in range(k):
        is_eval = (i * eval_size <= idxs) & (idxs < (i + 1) * eval_size)

        df_train = df.filter(~is_eval)
        df_eval = df.filter(is_eval)

        if as_lazy:
            folds[i] = {"train": df_train, "eval": df_eval}
        else:
            folds[i] = {"train": df_train.collect(), "eval": df_eval.collect()}

    if as_dict:
        return folds
    else:
        return [tuple(fold.values()) for fold in folds]


def _split_into_train_eval_k_folded(
    df: LazyFrame | DataFrame,
    eval_rel_size: float,
    k: int,
    stratify_by: Optional[str | List[str]] = None,
    shuffle: Optional[bool] = True,
    seed: Optional[int] = 273,
    as_lazy: Optional[bool] = False,
    validate: Optional[bool] = True,
    rel_size_deviation_tolerance: Optional[float] = 0.1,
@logger.catch
@enforce_input_outputs_expected_types
@validate_splitting
def _split_into_k_train_eval_folds(
    df: LazyFrame | DataFrame,
    eval_rel_size: float | None = None,
    k: int | None = 1,
    stratify_by: Optional[str | List[str]] = None,
    shuffle: Optional[bool] = True,
    seed: Optional[int] = 273,
    as_lazy: Optional[bool] = False,
    as_dict: Optional[bool] = False,
    validate: Optional[bool] = True,
    rel_size_deviation_tolerance: Optional[float] = 0.1,
) -> (
    Tuple[LazyFrame, LazyFrame]
    | Tuple[DataFrame, DataFrame]
    | List[Tuple[LazyFrame, LazyFrame]]
    | List[Tuple[DataFrame, DataFrame]]
    | List[Dict[str, LazyFrame]]
    | List[Dict[str, DataFrame]]
):
    """Split a DataFrame or LazyFrame into k non-overlapping folds, allowing for stratification by a column or list of columns."""

    idxs = int_range(0, count())
    if shuffle:
        idxs = idxs.shuffle(seed=seed)

    if k > 1:  # k-fold
        eval_rel_size = 1 / k

    eval_size = (eval_rel_size * count()).round(0).clip_min(1).cast(Int64)

    if stratify_by:
        idxs = idxs.over(stratify_by)
        eval_size = eval_size.over(stratify_by)

    folds = [{"train": None, "eval": None} for i in range(k)]
    for i in range(k):
        is_eval = i * eval_size <= idxs
        is_eval = is_eval & (idxs < (i + 1) * eval_size)

        folds[i] = {"train": df.filter(~is_eval), "eval": df.filter(is_eval)}

    return folds
