from typing import Dict, List, Literal, Tuple, Union, overload

from polars import DataFrame, Int64, LazyFrame, count, int_range

from polars_splitters.utils.validators import validate_inputs

df_pl = DataFrame | LazyFrame


@overload
def _split_into_subsets(
    as_dict: Literal[True],
    as_lazy: Literal[True],
    *args,
    **kwargs,
) -> Dict[str, LazyFrame]:
    ...


@overload
def _split_into_subsets(
    as_dict: Literal[True],
    as_lazy: Literal[False],
    *args,
    **kwargs,
) -> Dict[str, DataFrame]:
    ...


@overload
def _split_into_subsets(
    as_dict: Literal[False],
    as_lazy: Literal[True],
    *args,
    **kwargs,
) -> Dict[str, LazyFrame]:
    ...


@overload
def _split_into_subsets(
    as_dict: Literal[False],
    as_lazy: Literal[False],
    *args,
    **kwargs,
) -> Tuple[DataFrame, ...]:
    ...


def _split_into_subsets(
    df: df_pl,
    rel_sizes: Tuple[float, ...] | Dict[str, float],
    stratify_by: List[str] = None,
    shuffle: bool = True,
    as_dict: bool = False,
    as_lazy: bool = False,
    seed: int = 273,
) -> Tuple[df_pl, ...] | Dict[str, df_pl]:
    """Split a dataset into non-overlapping len(rel_sizes) subsets.

    Args:
        df (polars.DataFrame or polars.LazyFrame): The DataFrame to split.
        rel_sizes (tuple or dict): The relative sizes of the subsets. If a tuple, the sizes are interpreted as proportions
            of the total number of rows. If a dictionary, the keys are the names of the subsets and the values are the
            relative sizes. It must sum to 1.
        stratify_by (list of str, optional): The column names to use for stratification. If None (default), stratification is not
            performed. Note: Stratification by continuously-valued columns (float) is not currently supported.
        shuffle (bool, optional): Whether to shuffle the rows before splitting. Defaults to True.
        as_dict (bool, optional): Whether to return the subsets as a dictionary with the subset names as keys. Defaults
            to False.
        as_lazy (bool, optional): Whether to return the subsets as lazy DataFrames. Defaults to False.
        seed (int, optional): The random seed to use for shuffling. Defaults to 273.

    Returns:
        tuple or dict: The subsets as DataFrames. If `as_dict` is True, the subsets are returned as a dictionary with
            the subset names as keys.

    Raises:
        ValueError: If `rel_sizes` does not sum to one.
        NotImplementedError: If any column in `stratify_by` is of type float.
    """
    rel_sizes_ = rel_sizes
    if isinstance(rel_sizes, tuple):
        rel_sizes_ = {f"subset_{i}": rel_size for i, rel_size in enumerate(rel_sizes)}

    df_lazy = df.lazy()

    idxs = int_range(0, count())
    if shuffle:
        idxs = idxs.shuffle(seed=seed)

    sizes = {subset: (rel_size * count()).floor().cast(Int64) for subset, rel_size in rel_sizes_.items()}

    if stratify_by and isinstance(stratify_by, str):
        stratify_by = [stratify_by]

        idxs = idxs.over(stratify_by)
        sizes = {k: v.over(stratify_by) for k, v in sizes.items()}

    sizes = {k: v for k, v in sizes.items()}
    subsets = {subset: None for subset in sizes.keys()}
    for i, subset in enumerate(subsets.keys()):
        sizes_already_used = sum(list(sizes.values())[:i])
        is_subset = (sizes_already_used <= idxs) & (idxs < sizes_already_used + sizes[subset])
        subsets[subset] = df_lazy.filter(is_subset)

    if not as_lazy:
        subsets = {k: v.collect() for k, v in subsets.items()}

    if isinstance(rel_sizes, dict) and as_dict:
        return subsets
    else:
        return tuple(subsets.values())


@overload
def split_into_train_val_test(
    as_dict: Literal[True],
    as_lazy: Literal[True],
    *args,
    **kwargs,
) -> Dict[str, LazyFrame]:
    ...


@overload
def split_into_train_val_test(
    as_dict: Literal[True],
    as_lazy: Literal[False],
    *args,
    **kwargs,
) -> Dict[str, DataFrame]:
    ...


@overload
def split_into_train_val_test(
    as_dict: Literal[False],
    as_lazy: Literal[False],
    *args,
    **kwargs,
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    ...


@overload
def split_into_train_val_test(
    as_dict: Literal[False],
    as_lazy: Literal[True],
    *args,
    **kwargs,
) -> Tuple[LazyFrame, LazyFrame, LazyFrame]:
    ...


@validate_inputs
def split_into_train_val_test(
    df: df_pl,
    rel_sizes: Tuple[float, float, float],
    stratify_by: List[str] = None,
    shuffle: bool = True,
    as_dict: bool = False,
    as_lazy: bool = False,
    seed: int = 273,
) -> Tuple[df_pl, df_pl, df_pl] | Dict[str, df_pl]:
    """Split a dataset into non-overlapping train, validation, and test sets. Allows for stratification by a column or list of columns.

    This is a specific case of `split_into_subsets` in which the `rel_sizes` are limited to 3 and are interpreted as `(train_rel_size, val_rel_size, test_rel_size)`.

    Args:
        df (polars.DataFrame or polars.LazyFrame): The DataFrame to split.
        rel_sizes (tuple): A tuple of 3 rel_sizes representing the relative sizes of the subsets train, validation and test, respectively. The rel_sizes must sum to 1.
        stratify_by (list of str, optional): The column names to use for stratification. If None (default), stratification is not
            performed. Note: Stratification by continuously-valued columns (float) is not currently supported.
        shuffle (bool, optional): Whether to shuffle the rows before splitting. Defaults to True.
        as_dict (bool, optional): Whether to return the subsets as a dictionary with the subset names as keys. Defaults
            to False.
        as_lazy (bool, optional): Whether to return the subsets as lazy DataFrames. Defaults to False.
        seed (int, optional): The random seed to use for shuffling. Defaults to 273.

    Returns:
        tuple or dict: The subsets as DataFrames. If `as_dict` is True, the subsets are returned as a dictionary with
            the subset names as keys.

    Raises:
        ValueError: If `rel_sizes` does not sum to one.
        NotImplementedError: If any column in `stratify_by` is of type float, which is not currently supported.
    """

    subsets = _split_into_subsets(df, rel_sizes, stratify_by, shuffle, as_dict=False, as_lazy=as_lazy, seed=seed)
    if as_dict:
        return {k: v for k, v in zip(("train", "val", "test"), subsets)}
    else:
        return subsets


@overload
def split_into_train_eval(
    as_dict: Literal[True],
    as_lazy: Literal[True],
    *args,
    **kwargs,
) -> Dict[str, LazyFrame]:
    ...


@overload
def split_into_train_eval(
    as_dict: Literal[True],
    as_lazy: Literal[False],
    *args,
    **kwargs,
) -> Dict[str, DataFrame]:
    ...


@overload
def split_into_train_eval(
    as_dict: Literal[False],
    as_lazy: Literal[False],
    *args,
    **kwargs,
) -> Tuple[DataFrame, DataFrame]:
    ...


@overload
def split_into_train_eval(
    as_dict: Literal[False],
    as_lazy: Literal[True],
    *args,
    **kwargs,
) -> Tuple[LazyFrame, LazyFrame]:
    ...


@validate_inputs
def split_into_train_eval(
    df: df_pl,
    eval_rel_size: float,
    stratify_by: List[str] = None,
    shuffle: bool = True,
    as_dict: bool = False,
    as_lazy: bool = False,
    seed: int = 273,
) -> Tuple[df_pl, df_pl] | Dict[str, df_pl]:
    """Split a dataset into non-overlapping train and eval sets. Allows for stratification by a column or list of columns.

    This is a specific case of `split_into_subsets` in which the `rel_sizes` represent`(train_rel_size, eval_rel_size)`, with train_rel_size = 1 - eval_rel_size.

    Args:
        df (polars.DataFrame or polars.LazyFrame): The DataFrame to split.
        eval_rel_size (float): A decimal representing the relative size of the evaluation subset.
        stratify_by (list of str, optional): The column names to use for stratification. If None (default), stratification is not
            performed. Note: Stratification by continuously-valued columns (float) is not currently supported.
        shuffle (bool, optional): Whether to shuffle the rows before splitting. Defaults to True.
        as_dict (bool, optional): Whether to return the subsets as a dictionary with the subset names as keys. Defaults
            to False.
        as_lazy (bool, optional): Whether to return the subsets as lazy DataFrames. Defaults to False.
        seed (int, optional): The random seed to use for shuffling. Defaults to 273.

    Returns:
        tuple or dict: The subsets as DataFrames. If `as_dict` is True, the subsets are returned as a dictionary with
            the subset names as keys.

    Raises:
        NotImplementedError: If any column in `stratify_by` is of type float, which is not currently supported.
    """
    rel_sizes = (1 - eval_rel_size, eval_rel_size)
    subsets = _split_into_subsets(df, rel_sizes, stratify_by, shuffle, as_dict=False, as_lazy=as_lazy, seed=seed)
    if as_dict:
        return {k: v for k, v in zip(("train", "eval"), subsets)}
    else:
        return subsets


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
