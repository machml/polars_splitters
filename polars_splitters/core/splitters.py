from typing import Dict, List, Tuple, Union, overload

from loguru import logger
from polars import FLOAT_DTYPES, DataFrame, Int64, LazyFrame, count, int_range
from polars import selectors as cs

from polars_splitters.utils.type_enforcers import enforce_type

df_pl = DataFrame | LazyFrame

__all__ = [
    "split_into_train_eval",
    "split_into_k_folds",
]


def get_suggestion_for_loosening_stratification(func_name: str) -> str:
    if func_name == "split_into_train_eval":
        balance_suggestion = "using an `eval_rel_size` closer to 0.5"
    elif func_name == "split_into_k_folds":
        balance_suggestion = "using a smaller `k`"

    return f"""
            Consider:
            (a) {balance_suggestion},
            (b) using fewer columns in `stratify_by` columns,
            (c) disabling stratification altogether (stratify_by=None) or
            (d) using a larger input dataset df.
    """


def get_lazyframe_size(df: LazyFrame) -> int:
    return df.select(count()).collect().item()


@logger.catch
def split_into_train_eval(
    df_lazy: LazyFrame | DataFrame,
    eval_rel_size: float,
    stratify_by: str | List[str] = None,
    shuffle: bool = True,
    seed: int = 273,
    validate: bool = True,
    rel_size_deviation_tolerance: float = 0.1,
) -> Tuple[LazyFrame, LazyFrame]:
    r"""
    Split a dataset into non-overlapping train and eval sets, optionally stratifying by a column or list of columns.
    It is a wrapper around _split_into_train_eval, containing additional type coercion for the inputs, as well as validation for the inputs and outputs.

    Parameters
    ----------
    df_lazy : LazyFrame | DataFrame
        _description_
    eval_rel_size : float
        _description_
    stratify_by : str | List[str], optional. Defaults to None.
        _description_
    shuffle : bool, optional. Defaults to True.
        _description_
    seed : int, optional. Defaults to 273.
        _description_
    validate : bool, optional. Defaults to True.
        _description_
    rel_size_deviation_tolerance : float, optional. Defaults to 0.1.
        _description_

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
    >>> df = pl.DataFrame(
    ...     {
    ...         "feature_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ...         "treatment": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    ...         "outcome":   [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    ...     }
    ... )
    >>> df_train, df_test = split_into_train_eval(df, eval_rel_size=0.3, stratify_by=["treatment", "outcome"], shuffle=True)
    >>> print(df_train.collect(), df_test.collect(), sep="\n\n")
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
    │ 6.0       ┆ 1         ┆ 0       │
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
    │ 7.0       ┆ 1         ┆ 0       │
    │ 10.0      ┆ 1         ┆ 1       │
    └───────────┴───────────┴─────────┘
    """
    # type coercion
    df_lazy = df_lazy.lazy()
    stratify_by = enforce_type(stratify_by, list)

    if validate:
        validate_eval_rel_size_setting(eval_rel_size)

        if stratify_by:
            # validate stratification dtypes
            stratification_columns_of_float_type = df_lazy.select(stratify_by).select(cs.by_dtype(FLOAT_DTYPES)).schema
            if stratification_columns_of_float_type:
                raise NotImplementedError(
                    f"""
                    Attempted to stratify based on float column(s): {stratification_columns_of_float_type}.
                    This is not currently supported. Consider discretizing the column first or using a different column.
                    """
                )

            # validate stratification feasibility (size_input, eval_rel_size, n_strata, stratify_by)
            input_size = get_lazyframe_size(df_lazy)
            eval_size = df_lazy.select((eval_rel_size * count()).floor().clip_min(1).cast(Int64)).collect().item()
            if eval_rel_size <= 0.5:
                smallest_set, smallest_set_size = ("eval", eval_size)
            else:
                smallest_set, smallest_set_size = ("train", input_size - eval_size)

            n_strata = df_lazy.select(stratify_by).collect().n_unique()
            logger.debug(
                f"input_size: {input_size}, eval_size: {eval_size}, (input_size - eval_size): {input_size - eval_size}"
            )
            logger.debug(
                f"smallest_set: {smallest_set}, smallest_set_size: {smallest_set_size}, n_strata: {n_strata}, stratify_by: {stratify_by}"
            )
            if smallest_set_size < n_strata:
                f"""
                Unable to generate the data splits for the data df and the configuration attempted for eval_rel_size and stratify_by.
                For the stratification to work, the size of the smallest set (currently {smallest_set}: {smallest_set_size})
                must be at least as large as the number of strata (currently {n_strata}), i.e. the number of unique row-wise combinations of values in the stratify_by columns (currently {stratify_by}).

                {get_suggestion_for_loosening_stratification("split_into_train_eval")}
                """

    df_train, df_eval = _split_into_train_eval(df_lazy, eval_rel_size, stratify_by, shuffle, seed)

    if validate:
        eval_rel_size_actual = get_lazyframe_size(df_eval) / input_size

        logger.debug(f"eval_rel_size: {eval_rel_size}, eval_rel_size_actual: {eval_rel_size_actual}")

        if abs(eval_rel_size_actual - eval_rel_size) > rel_size_deviation_tolerance:
            raise ValueError(
                f"""
                The actual relative size of the eval set ({eval_rel_size_actual}) deviates from the requested relative size ({eval_rel_size}) by more than the specified tolerance ({rel_size_deviation_tolerance}).

                {get_suggestion_for_loosening_stratification("split_into_train_eval")}
                """
            )

    return (df_train, df_eval)


def _split_into_train_eval(
    df_lazy: LazyFrame,
    eval_rel_size: float,
    stratify_by: List[str] = None,
    shuffle: bool = True,
    seed: int = 273,
) -> Tuple[LazyFrame, LazyFrame]:
    """Split a dataset into non-overlapping train and eval sets. Allows for stratification by a column or list of columns."""
    idxs = int_range(0, count())
    if shuffle:
        idxs = idxs.shuffle(seed=seed)

    eval_size = (eval_rel_size * count()).floor().clip_min(1).cast(Int64)

    if stratify_by:
        idxs = idxs.over(stratify_by)
        eval_size = eval_size.over(stratify_by)

    is_eval = idxs < eval_size

    df_train = df_lazy.filter(~is_eval)
    df_eval = df_lazy.filter(is_eval)

    return (df_train, df_eval)


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


def k_fold(df: df_pl, k: int, shuffle: bool = True, seed: int = 273):
    return split_into_k_folds(df, k, stratify_by=None, shuffle=shuffle, as_dict=False, as_lazy=False, seed=seed)


get_k_folds = split_into_k_folds
stratified_k_fold = split_into_k_folds
train_test_split = split_into_train_eval
train_val_test_split = split_into_train_val_test
