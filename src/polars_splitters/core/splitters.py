from typing import Literal, overload

from loguru import logger
from polars import DataFrame, Int64, LazyFrame, col, int_range
from polars import len as pl_len
from polars import selectors as cs

from polars_splitters.utils.guardrails import (
    enforce_input_outputs_expected_types,
    validate_splitting,
)

df_pl = DataFrame | LazyFrame

__all__ = [
    "split_into_k_folds",
    "split_into_train_eval",
    "sample",
]


def split_into_k_folds(
    df: LazyFrame | DataFrame,
    k: int | None = 1,
    stratify_by: str | list[str] | None = None,
    float_qbins: int | dict[str, int] = 10,
    shuffle: bool | None = True,
    seed: int | None = 173,
    as_lazy: bool | None = False,
    as_dict: bool | None = False,
    validate: bool | None = True,
    rel_size_deviation_tolerance: float | None = 0.1,
) -> (
    list[tuple[LazyFrame, LazyFrame]]
    | list[tuple[DataFrame, DataFrame]]
    | list[dict[str, LazyFrame]]
    | list[dict[str, DataFrame]]
):
    """Split a DataFrame or LazyFrame into k non-overlapping folds, allowing for stratification by a column or list of columns."""
    return _split_into_k_train_eval_folds(
        df=df,
        eval_rel_size=None,
        k=k,
        stratify_by=stratify_by,
        float_qbins=float_qbins,
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
    k: int | None = 1,
    stratify_by: str | list[str] | None = None,
    float_qbins: int | dict[str, int] = 10,
    shuffle: bool | None = True,
    seed: int | None = 173,
    as_lazy: bool | None = False,
    as_dict: bool | None = False,
    validate: bool | None = True,
    rel_size_deviation_tolerance: float | None = 0.1,
) -> (
    tuple[LazyFrame, LazyFrame]
    | tuple[DataFrame, DataFrame]
    | dict[str, LazyFrame]
    | dict[str, DataFrame]
    | list[tuple[LazyFrame, LazyFrame]]
    | list[tuple[DataFrame, DataFrame]]
    | list[dict[str, LazyFrame]]
    | list[dict[str, DataFrame]]
): ...


@overload
def _split_into_k_train_eval_folds(
    df: LazyFrame | DataFrame,
    eval_rel_size: float | None = None,
    k: int | None = 1,
    stratify_by: str | list[str] | None = None,
    float_qbins: int | dict[str, int] = 10,
    shuffle: bool | None = True,
    seed: int | None = 173,
    as_lazy: bool | None = False,
    as_dict: bool | None = False,
    validate: bool | None = True,
    rel_size_deviation_tolerance: float | None = 0.1,
) -> (
    tuple[LazyFrame, LazyFrame]
    | tuple[DataFrame, DataFrame]
    | list[tuple[LazyFrame, LazyFrame]]
    | list[tuple[DataFrame, DataFrame]]
    | list[dict[str, LazyFrame]]
    | list[dict[str, DataFrame]]
): ...


@overload
def _split_into_k_train_eval_folds(
    df: LazyFrame | DataFrame,
    eval_rel_size: float | None = None,
    k: int | None = 1,
    stratify_by: str | list[str] | None = None,
    float_qbins: int | dict[str, int] = 10,
    shuffle: bool | None = True,
    seed: int | None = 173,
    as_lazy: bool | None = False,
    as_dict: bool | None = False,
    validate: bool | None = True,
    rel_size_deviation_tolerance: float | None = 0.1,
) -> (
    tuple[LazyFrame, LazyFrame]
    | tuple[DataFrame, DataFrame]
    | list[tuple[LazyFrame, LazyFrame]]
    | list[tuple[DataFrame, DataFrame]]
    | list[dict[str, LazyFrame]]
    | list[dict[str, DataFrame]]
): ...


@overload
def _split_into_k_train_eval_folds(
    df: LazyFrame | DataFrame,
    eval_rel_size: float | None = None,
    k: int | None = 1,
    stratify_by: str | list[str] | None = None,
    float_qbins: int | dict[str, int] = 10,
    shuffle: bool | None = True,
    seed: int | None = 173,
    as_lazy: bool | None = False,
    as_dict: bool | None = False,
    validate: bool | None = True,
    rel_size_deviation_tolerance: float | None = 0.1,
) -> (
    tuple[LazyFrame, LazyFrame]
    | tuple[DataFrame, DataFrame]
    | list[tuple[LazyFrame, LazyFrame]]
    | list[tuple[DataFrame, DataFrame]]
    | list[dict[str, LazyFrame]]
    | list[dict[str, DataFrame]]
): ...


@logger.catch
@enforce_input_outputs_expected_types
@validate_splitting
def _split_into_k_train_eval_folds(
    df: LazyFrame | DataFrame,
    eval_rel_size: float | None = None,
    k: int = 1,
    stratify_by: str | list[str] | None = None,
    float_qbins: int | dict[str, int] = 10,
    shuffle: bool | None = True,
    seed: int | None = 173,
    as_lazy: bool | None = False,
    as_dict: bool | None = False,
    validate: bool | None = True,
    rel_size_deviation_tolerance: float | None = 0.1,
) -> (
    tuple[LazyFrame, LazyFrame]
    | tuple[DataFrame, DataFrame]
    | list[tuple[LazyFrame, LazyFrame]]
    | list[tuple[DataFrame, DataFrame]]
    | list[dict[str, LazyFrame]]
    | list[dict[str, DataFrame]]
):
    """Split a DataFrame or LazyFrame into k non-overlapping folds, allowing for stratification by a column or list of columns."""
    idxs = int_range(0, pl_len())
    if shuffle:
        idxs = idxs.shuffle(seed=seed)

    if k > 1:  # k-fold
        eval_rel_size = 1 / k

    eval_size = (eval_rel_size * pl_len()).round(0).clip(lower_bound=1).cast(Int64)

    df_preprocessed = df.clone()
    if stratify_by:
        float_strat_cols = df.select(cs.float() & cs.by_name(stratify_by)).columns

        if float_strat_cols:
            logger.info(
                f"Float columns found among stratify_by columns: {float_strat_cols}. Discretizing these columns into {float_qbins} quantile bins",
            )
            if isinstance(float_qbins, int):
                float_qbins = dict.fromkeys(float_strat_cols, float_qbins)

            df_preprocessed = df.with_columns(
                [
                    col(col_name).qcut(float_qbins[col_name]).alias(f"polars_splitters:{col_name}:qcut")
                    for col_name in float_strat_cols
                ],
            )

            stratify_by = [
                f"polars_splitters:{col_name}:qcut" if col_name in float_strat_cols else col_name
                for col_name in stratify_by
            ]
            logger.debug(f"stratify_by: {stratify_by}")

        idxs = idxs.over(stratify_by)
        eval_size = eval_size.over(stratify_by)

    folds = [{"train": None, "eval": None} for i in range(k)]
    for i in range(k):
        is_eval = i * eval_size <= idxs
        is_eval = is_eval & (idxs < (i + 1) * eval_size)

        folds[i] = {
            "train": df_preprocessed.filter(~is_eval).select(df.columns),
            "eval": df_preprocessed.filter(is_eval).select(df.columns),
        }

    return folds


def split_into_train_eval(
    df: LazyFrame | DataFrame,
    eval_rel_size: float,
    stratify_by: str | list[str] | None = None,
    float_qbins: int | dict[str, int] = 10,
    shuffle: bool | None = True,
    seed: int | None = 173,
    as_lazy: bool | None = False,
    as_dict: bool | None = False,
    validate: bool | None = True,
    rel_size_deviation_tolerance: float | None = 0.1,
) -> tuple[LazyFrame, LazyFrame] | tuple[DataFrame, DataFrame] | dict[str, LazyFrame] | dict[str, DataFrame]:
    r"""Split a dataset into non-overlapping train and eval sets, optionally stratifying by a column or list of columns.
    It includes logging and some guardrails: type coercion as well as validation for the inputs and outputs.

    Parameters
    ----------
    df : LazyFrame | DataFrame
        The polars DataFrame to split.
    eval_rel_size : float
        The targeted relative size of the eval set. Must be between 0.0 and 1.0.
    stratify_by : str | list[str], optional. Defaults to None.
        The column names to use for stratification.
        If None (default), stratification is not performed. Note: Stratification by float columns is not currently supported.
    float_qbins : int | dict[str, int], optional. Defaults to 10 (deciles).
        How many quantile bins should be used for discretizing float-typed columns in stratify_by, e.g., 10 for discretizing in deciles (default), 5 for quintiles.
        Can be specified as a constant to be used across all float-typed columns in stratify_by, or as a dictionary in the format {<float_col_name>:<float_qbins>}.
        If no float-typed column in stratify_by, this is ignored.
    shuffle : bool, optional. Defaults to True.
        Whether to shuffle the rows before splitting.
    seed : int, optional. Defaults to 173.
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
    tuple[LazyFrame, LazyFrame] | tuple[DataFrame, DataFrame] | dict[str, LazyFrame] | dict[str, DataFrame]
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
    ...         "outcome": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    ...     }
    ... )
    >>> df_train, df_eval = split_into_train_eval(
    ...     df, eval_rel_size=0.3, stratify_by=["treatment", "outcome"], shuffle=True, as_lazy=False
    ... )
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
        float_qbins=float_qbins,
        shuffle=shuffle,
        seed=seed,
        as_lazy=as_lazy,
        as_dict=as_dict,
        validate=validate,
        rel_size_deviation_tolerance=rel_size_deviation_tolerance,
    )


def sample(
    df: DataFrame,
    fraction: float,
    stratify_by: str | list[str],
    float_qbins: int | dict[str, int] = 10,
    fraction_rel_tolerance: float | None = 0.1,
    seed: int = 173,
) -> DataFrame:
    """
    Get a stratified sample from a polars DataFrame.

    Parameters
    ----------
    df : DataFrame
        Data to be sampled.
    fraction : float
        A number from 0.0 to 1.0 specifying the size of the sample relative to the original dataframe.
    stratify_by : str | list[str]
        Column(s) to use for stratification.
    float_qbins : int | dict[str, int], optional. Defaults to 10 (deciles).
        How many quantile bins should be used for discretizing float-typed columns in stratify_by, e.g., 10 for discretizing in deciles (default), 5 for quintiles.
        Can be specified as a constant to be used across all float-typed columns in stratify_by, or as a dictionary in the format {<float_col_name>:<float_qbins>}.
        If no float-typed column in stratify_by, this is ignored.
    fraction_rel_tolerance : float, optional. Defaults to 0.1.
        Sets the maximum allowed abs(fraction_actual - fraction_size).
        When stratifying, the fraction_actual might deviate from the targeted fraction_size due to the fact that strata for the given data may not be perfectly divisible at the desired proportion (eval_rel_size * df.height is not integer).
        If validate is set to False, this parameter is ignored.
    seed : int, optional. Defaults to 173.
        The random seed to use in shuffling.

    Returns
    -------
    DataFrame
        Stratified sample.
    """
    # _, df_sample = split_into_train_eval(
    #     df,
    #     eval_rel_size=fraction,
    #     stratify_by=stratify_by,
    #     float_qbins=float_qbins,
    #     shuffle=True,
    #     as_lazy=False,
    #     seed=seed,
    # )
    # return df_sample
    _, df_sample = _split_into_k_train_eval_folds(
        df=df,
        eval_rel_size=fraction,
        k=1,
        stratify_by=stratify_by,
        float_qbins=float_qbins,
        shuffle=True,
        seed=seed,
        as_lazy=False,
        as_dict=False,
        validate=True,
        rel_size_deviation_tolerance=fraction_rel_tolerance,
    )

    return df_sample