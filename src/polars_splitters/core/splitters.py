from loguru import logger
from polars import DataFrame, Int64, LazyFrame, col, int_range
from polars import len as pl_len
from polars import selectors as cs

from polars_splitters.utils.guardrails import enforce_input_outputs_expected_types, validate_splitting

df_pl = DataFrame | LazyFrame

__all__ = [
    "split_into_k_folds",
    "split_into_train_eval",
    "sample",
]

TrainEvalTuple = tuple[DataFrame, DataFrame]
TrainEvalDict = dict[str, DataFrame]
LazyTrainEvalTuple = tuple[LazyFrame, LazyFrame]
LazyTrainEvalDict = dict[str, LazyFrame]


def split_into_k_folds(
    df: LazyFrame | DataFrame,
    k: int | None = 1,
    stratify_by: str | list[str] | None = None,
    max_numeric_cardinality: int | None = 20,
    numeric_high_cardinal_qbins: int | dict[str, int] = 10,
    shuffle: bool | None = True,
    seed: int | None = 173,
    as_lazy: bool | None = False,
    as_dict: bool | None = False,
    validate: bool | None = True,
    rel_size_deviation_tolerance: float | None = 0.1,
) -> list[LazyTrainEvalTuple] | list[TrainEvalTuple] | list[LazyTrainEvalDict] | list[TrainEvalDict]:
    """Split a DataFrame or LazyFrame into k non-overlapping folds, allowing for stratification by a column or list of columns."""
    return _split_into_k_train_eval_folds(
        df=df,
        eval_rel_size=None,
        k=k,
        stratify_by=stratify_by,
        max_numeric_cardinality=max_numeric_cardinality,
        numeric_high_cardinal_qbins=numeric_high_cardinal_qbins,
        shuffle=shuffle,
        seed=seed,
        as_lazy=as_lazy,
        as_dict=as_dict,
        validate=validate,
        rel_size_deviation_tolerance=rel_size_deviation_tolerance,
    )


@logger.catch
@enforce_input_outputs_expected_types
@validate_splitting
def _split_into_k_train_eval_folds(
    df: LazyFrame | DataFrame,
    k: int = 1,
    stratify_by: str | list[str] | None = None,
    max_numeric_cardinality: int = 20,
    numeric_high_cardinal_qbins: int | dict[str, int] = 10,
    shuffle: bool = True,
    seed: int = 173,
    validate: bool = True,
    rel_size_deviation_tolerance: float = 0.1,
) -> list[TrainEvalDict]:
    """
    Split a DataFrame or LazyFrame into k non-overlapping folds, allowing for stratification by a column or list of columns.
    Always returns a list of dicts with keys 'train' and 'eval', each containing a DataFrame.
    """
    idxs = int_range(0, pl_len())
    if shuffle:
        idxs = idxs.shuffle(seed=seed)

    if k > 1:  # k-fold
        eval_rel_size = 1 / k

    eval_size = (eval_rel_size * pl_len()).round(0).clip(lower_bound=1).cast(Int64)

    df_preprocessed = df.clone()
    if stratify_by:
        strat_nums = df.select(cs.numeric() & cs.by_name(stratify_by)).columns
        if len(strat_nums) > 0:
            if max_numeric_cardinality is None:
                max_numeric_cardinality = int(1e6)
            df_ = df.collect() if isinstance(df, LazyFrame) else df.clone()
            high_cardinality_num_strat_cols = [
                col_name for col_name in strat_nums if df_[col_name].n_unique() > max_numeric_cardinality
            ]
            df_.clear()

            if len(high_cardinality_num_strat_cols) > 0:
                logger.info(
                    f"""Numeric columns with high cardinality (>{max_numeric_cardinality} uniques) found among stratify_by columns: {high_cardinality_num_strat_cols}.
                    Its quantilized version ({numeric_high_cardinal_qbins} quantile bins) will be use for stratification instead."""
                )
                if isinstance(numeric_high_cardinal_qbins, int):
                    numeric_high_cardinal_qbins = dict.fromkeys(
                        high_cardinality_num_strat_cols, numeric_high_cardinal_qbins
                    )

                df_preprocessed = df.with_columns(
                    [
                        col(col_name)
                        .qcut(numeric_high_cardinal_qbins[col_name])
                        .alias(f"polars_splitters:{col_name}:qcut")
                        for col_name in high_cardinality_num_strat_cols
                    ],
                )

                stratify_by = [
                    f"polars_splitters:{col_name}:qcut" if col_name in high_cardinality_num_strat_cols else col_name
                    for col_name in stratify_by
                ]
                logger.debug(f"stratify_by: {stratify_by}")

        idxs = idxs.over(stratify_by)
        eval_size = eval_size.over(stratify_by)

    folds: list[TrainEvalDict] = []
    for i in range(k):
        is_eval = i * eval_size <= idxs
        is_eval = is_eval & (idxs < (i + 1) * eval_size)

        df_train = df_preprocessed.filter(~is_eval).select(df.columns)
        df_eval = df_preprocessed.filter(is_eval).select(df.columns)

        df_train = df_train.collect() if isinstance(df_train, LazyFrame) else df_train
        df_eval = df_eval.collect() if isinstance(df_eval, LazyFrame) else df_eval

        folds.append({"train": df_train, "eval": df_eval})

    return folds


def split_into_train_eval(
    df: LazyFrame | DataFrame,
    eval_rel_size: float,
    stratify_by: str | list[str] | None = None,
    max_numeric_cardinality: int | None = 20,
    numeric_high_cardinal_qbins: int | dict[str, int] = 5,
    shuffle: bool | None = True,
    seed: int | None = 173,
    validate: bool | None = True,
    rel_size_deviation_tolerance: float | None = 0.1,
) -> TrainEvalTuple:
    r"""
    Split a dataset into non-overlapping train and eval sets, optionally stratifying by a column or list of columns.
    Includes logging and guardrails: type coercion and validation for inputs and outputs.

    Parameters
    ----------
    df : LazyFrame | DataFrame
        The polars DataFrame to split.
    eval_rel_size : float
        Targeted relative size of the eval set. Must be between 0.0 and 1.0.
    stratify_by : str | list[str], optional
        Column name(s) to use for stratification. If None, stratification is not performed. Stratification by float columns is not supported.
    max_numeric_cardinality : int, optional
        Maximum unique values allowed in numeric columns for direct stratification. Columns exceeding this will be quantile-binned. Defaults to 20.
    numeric_high_cardinal_qbins : int | dict[str, int], optional
        Number of quantile bins for high-cardinality numeric columns in stratify_by. Can be an int (applies to all) or dict mapping column names to bin counts. Defaults to 5.
    shuffle : bool, optional
        Whether to shuffle rows before splitting. Defaults to True.
    seed : int, optional
        Random seed for shuffling. Defaults to 173.
    validate : bool, optional
        Whether to validate inputs and outputs. Defaults to True.
    rel_size_deviation_tolerance : float, optional
        Maximum allowed absolute deviation between actual and requested eval set size. Defaults to 0.1.

    Returns
    -------
    tuple[DataFrame, DataFrame]
        Train and eval sets as DataFrames.

    Raises
    ------
    NotImplementedError
        If stratifying by a float column.
    ValueError
        If the actual eval set size deviates from the requested size by more than the specified tolerance, or if the smallest set is smaller than the number of strata.

    Examples
    --------
    >>> import polars as pl
    >>> from polars_splitters.core.splitters import split_into_train_eval
    >>> df = pl.DataFrame(
    ...     {
    ...         "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    ...         "treatment": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    ...         "outcome": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    ...     }
    ... )
    >>> df_train, df_eval = split_into_train_eval(
    ...     df, eval_rel_size=0.3, stratify_by=["treatment", "outcome"], shuffle=True
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
    df_train, df_eval = _split_into_k_train_eval_folds(
        df=df,
        eval_rel_size=eval_rel_size,
        k=1,
        stratify_by=stratify_by,
        max_numeric_cardinality=max_numeric_cardinality,
        numeric_high_cardinal_qbins=numeric_high_cardinal_qbins,
        shuffle=shuffle,
        seed=seed,
        as_lazy=False,
        as_dict=False,
        validate=validate,
        rel_size_deviation_tolerance=rel_size_deviation_tolerance,
    )

    assert isinstance(df_train, DataFrame) and isinstance(df_eval, DataFrame)
    return df_train, df_eval


def sample(
    df: DataFrame,
    fraction: float,
    stratify_by: str | list[str],
    max_numeric_cardinality: int | None = 20,
    numeric_high_cardinal_qbins: int | dict[str, int] = 5,
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
    float_qbins : int | dict[str, int], optional. Defaults to 5 (quintiles).
        How many quantile bins should be used for discretizing float-typed columns in stratify_by, e.g., 10 for discretizing in deciles, 5 for quintiles.
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
    _, df_sample = split_into_train_eval(
        df,
        eval_rel_size=fraction,
        stratify_by=stratify_by,
        max_numeric_cardinality=max_numeric_cardinality,
        rel_size_deviation_tolerance=fraction_rel_tolerance,
        numeric_high_cardinal_qbins=numeric_high_cardinal_qbins,
        shuffle=True,
        seed=seed,
    )
    return df_sample
