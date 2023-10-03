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
    df_lazy: LazyFrame | DataFrame,
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
    It is a wrapper around _split_into_train_eval including logging, pre- and post-processing and some guarding rails: type coercion and validation for the inputs and outputs.
    FIXME: not based on _split_into_train_eval anymore.

    Parameters
    ----------
    df_lazy : LazyFrame | DataFrame
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

    df_train = df_lazy.filter(~is_eval)
    df_eval = df_lazy.filter(is_eval)

    return df_train, df_eval


def split_into_k_folds(
    df: Union[DataFrame, LazyFrame],
    k: int,
    stratify_by: Union[str, List[str], Dict[str, float]] = None,
    shuffle: bool = True,
    as_dict: bool = True,
    as_lazy: bool = False,
    seed: int = 273,
) -> Tuple[df_pl, df_pl] | Dict[str, df_pl]:
    """Split a DataFrame or LazyFrame into k non-overlapping folds.
    Allows for stratification by a column or list of columns.
    If a single column is provided, all folds will have the same proportion of that column's values between train and eval sets.
    If a list of columns is provided, all folds will have the same proportion of combinations of those columns' values between train and eval sets.

    Notes:
    - Stratification by continuously-valued columns (float) is not currently supported.
    - All k folds have the same sizes for the train and eval sets. However, len(df) % k rows will never appear in the eval sets. This is usually negligible for most use cases, since typically (len(df) % k) / len(df) << 1.0.


    Args:
        df (polars.DataFrame or polars.LazyFrame): The DataFrame to split.
        k (int): The number of folds to create.
        stratify_by (str or list of str, optional): A column or list of columns to stratify the folds by.
        shuffle (bool, optional): Whether to shuffle the data before (stratifying and) splitting. Defaults to True.
        as_dict (bool, optional): Whether to return the folds as a list of dictionaries of the form
            {"train": ..., "eval": ...} (as_dict=True) or as a list of tuples (df_train, df_eval). Defaults to False.
        as_lazy (bool, optional): Whether to return the folds as LazyFrames or DataFrames. Defaults to False.
        seed (int, optional): The random seed to use for shuffling. Defaults to None.

    Returns:
        A dictionary of folds, where the keys are the names of the evaluation sets
        (e.g. "train", "valid", "test") and the values are the corresponding
        DataFrames or LazyFrames.

    Raises:
        ValueError: If k is less than 2.
        NotImplementedError: If any column in stratify_by is of type float.
    """
    if k < 2:
        raise ValueError("k must be greater than 1")

    df_lazy = df.lazy()

    folds = [{"train": None, "eval": None} for i in range(k)]
    for i in range(k):
        folds[i] = split_into_train_eval(
            df_lazy, 1 / k, stratify_by=stratify_by, shuffle=shuffle, as_dict=True, as_lazy=as_lazy, seed=seed
        )

    if as_dict:
        return folds
    else:
        return [tuple(fold.values()) for fold in folds]
