from typing import Dict, List, Literal, Optional, Tuple, overload

from loguru import logger
from polars import DataFrame, Int64, LazyFrame, int_range
from polars import len as pl_len

from polars_splitters.utils.guardrails import (
    enforce_input_outputs_expected_types,
    validate_splitting,
)

df_pl = DataFrame | LazyFrame

__all__ = [
    "split_into_train_eval",
    "split_into_k_folds",
]


def split_into_train_eval(
    df: LazyFrame | DataFrame,
    eval_rel_size: float,
    stratify_by: Optional[str | List[str]] = None,
    shuffle: Optional[bool] = True,
    seed: Optional[int] = 273,
    as_lazy: Optional[bool] = False,
    as_dict: Optional[bool] = False,
    validate: Optional[bool] = True,
    rel_size_deviation_tolerance: Optional[float] = 0.1,
) -> Tuple[LazyFrame, LazyFrame] | Tuple[DataFrame, DataFrame] | Dict[str, LazyFrame] | Dict[str, DataFrame]:
    r"""
    Split a dataset into non-overlapping train and eval sets, optionally stratifying by a column or list of columns.
    It includes logging and some guardrails: type coercion as well as validation for the inputs and outputs.

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
    as_dict : bool, optional. Defaults to False.
        Whether to return the train and eval sets as a tuple (False) or as a dictionary (True).
    validate : bool, optional. Defaults to True.
        Whether to validate the inputs and outputs.
    rel_size_deviation_tolerance : float, optional. Defaults to 0.1.
        Sets the maximum allowed abs(eval_rel_size_actual - eval_rel_size).
        When stratifying, the eval_rel_size_actual might deviate from eval_rel_size due to the fact that strata for the given data may not be perfectly divisible at the desired proportion (1-eval_rel_size, eval_rel_size).
        If validate is set to False, this parameter is ignored.

    Returns
    -------
    Tuple[LazyFrame, LazyFrame] | Tuple[DataFrame, DataFrame] | Dict[str, LazyFrame] | Dict[str, DataFrame]
        df_train and df_eval, either as a tuple or as a dictionary, and either as LazyFrames or DataFrames, depending on the values of as_dict and as_lazy.

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
    return _split_into_k_train_eval_folds(
        df=df,
        eval_rel_size=eval_rel_size,
        k=1,
        stratify_by=stratify_by,
        shuffle=shuffle,
        seed=seed,
        as_lazy=as_lazy,
        as_dict=as_dict,
        validate=validate,
        rel_size_deviation_tolerance=rel_size_deviation_tolerance,
    )


def split_into_k_folds(
    df: LazyFrame | DataFrame,
    k: Optional[int] = 1,
    stratify_by: Optional[str | List[str]] = None,
    shuffle: Optional[bool] = True,
    seed: Optional[int] = 273,
    as_lazy: Optional[bool] = False,
    as_dict: Optional[bool] = False,
    validate: Optional[bool] = True,
    rel_size_deviation_tolerance: Optional[float] = 0.1,
) -> (
    List[Tuple[LazyFrame, LazyFrame]]
    | List[Tuple[DataFrame, DataFrame]]
    | List[Dict[str, LazyFrame]]
    | List[Dict[str, DataFrame]]
):
    """Split a DataFrame or LazyFrame into k non-overlapping folds, allowing for stratification by a column or list of columns."""

    return _split_into_k_train_eval_folds(
        df=df,
        eval_rel_size=None,
        k=k,
        stratify_by=stratify_by,
        shuffle=shuffle,
        seed=seed,
        as_lazy=as_lazy,
        as_dict=as_dict,
        validate=validate,
        rel_size_deviation_tolerance=rel_size_deviation_tolerance,
    )


@overload
def _split_into_k_train_eval_folds(
    df: LazyFrame | DataFrame,
    eval_rel_size: Literal[None] = ...,
    k: Optional[int] = 1,
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
    | Dict[str, LazyFrame]
    | Dict[str, DataFrame]
    | List[Tuple[LazyFrame, LazyFrame]]
    | List[Tuple[DataFrame, DataFrame]]
    | List[Dict[str, LazyFrame]]
    | List[Dict[str, DataFrame]]
):
    ...


@overload
def _split_into_k_train_eval_folds(
    df: LazyFrame | DataFrame,
    eval_rel_size: float | None = None,
    k: Optional[int] = 1,
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
    ...


@overload
def _split_into_k_train_eval_folds(
    df: LazyFrame | DataFrame,
    eval_rel_size: float | None = None,
    k: Optional[int] = 1,
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
    ...


@overload
def _split_into_k_train_eval_folds(
    df: LazyFrame | DataFrame,
    eval_rel_size: float | None = None,
    k: Optional[int] = 1,
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
    ...


@logger.catch
@enforce_input_outputs_expected_types
@validate_splitting
def _split_into_k_train_eval_folds(
    df: LazyFrame | DataFrame,
    eval_rel_size: float | None = None,
    k: Optional[int] = 1,
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

    idxs = int_range(0, pl_len())
    if shuffle:
        idxs = idxs.shuffle(seed=seed)

    if k > 1:  # k-fold
        eval_rel_size = 1 / k

    eval_size = (eval_rel_size * pl_len()).round(0).clip(lower_bound=1).cast(Int64)

    if stratify_by:
        idxs = idxs.over(stratify_by)
        eval_size = eval_size.over(stratify_by)

    folds = [{"train": None, "eval": None} for i in range(k)]
    for i in range(k):
        is_eval = i * eval_size <= idxs
        is_eval = is_eval & (idxs < (i + 1) * eval_size)

        folds[i] = {"train": df.filter(~is_eval), "eval": df.filter(is_eval)}

    return folds
