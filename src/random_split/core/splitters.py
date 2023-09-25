import math
from functools import wraps
from typing import Dict, List, Literal, Tuple, Union, overload

import polars as pl

df_pl = pl.DataFrame | pl.LazyFrame
eval_set_name = Literal["val", "test"]


def validate_rel_sizes(func):
    """Assumes that rel_sizes is the second argument of the function being wrapped."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        rel_sizes_ = kwargs.get("rel_sizes", None)
        if rel_sizes_ is None:
            rel_sizes_ = args[1] if args else None

        if isinstance(rel_sizes_, dict):
            rel_sizes_ = tuple(rel_sizes_.values())

        # sum should be 1
        if not math.isclose(sum(rel_sizes_), 1.0, abs_tol=1e-6):
            raise ValueError("rel_sizes must sum to 1")

        # all values should be non-negative
        if any(rel_size < 0 for rel_size in rel_sizes_):
            raise ValueError("rel_sizes must be non-negative")

        # all values should be less than 1
        if any(rel_size > 1 for rel_size in rel_sizes_):
            raise ValueError("rel_sizes must be less than 1")

        # rel_sizes should be of type Tuple[float], Dict[Any, float]
        if not isinstance(rel_sizes_, (tuple, dict)):
            raise TypeError("rel_sizes must be of type Tuple[float], Dict[Any, float]")
        if isinstance(rel_sizes_, tuple) and not all(isinstance(rel_size, float) for rel_size in rel_sizes_):
            raise TypeError("rel_sizes must be of type Tuple[float], Dict[Any, float]")
        if isinstance(rel_sizes_, dict) and not all(isinstance(rel_size, float) for rel_size in rel_sizes_.values()):
            raise TypeError("rel_sizes must be of type Tuple[float], Dict[Any, float]")

        return func(*args, **kwargs)

    return wrapper


@overload
def split_into_subsets(
    as_dict: Literal[True],
    as_lazy: Literal[True],
    *args,
    **kwargs,
) -> Dict[eval_set_name, pl.LazyFrame]:
    ...


@overload
def split_into_subsets(
    as_dict: Literal[True],
    as_lazy: Literal[False],
    *args,
    **kwargs,
) -> Dict[eval_set_name, pl.DataFrame]:
    ...


@overload
def split_into_subsets(
    as_dict: Literal[False],
    as_lazy: Literal[True],
    *args,
    **kwargs,
) -> Dict[eval_set_name, pl.LazyFrame]:
    ...


@overload
def split_into_subsets(
    as_dict: Literal[False],
    as_lazy: Literal[False],
    *args,
    **kwargs,
) -> Tuple[pl.DataFrame, ...]:
    ...


@validate_rel_sizes
def split_into_subsets(
    df: df_pl,
    rel_sizes: Tuple[float, ...] | Dict[str, float],
    stratify_by: List[str] = None,
    shuffle: bool = True,
    as_dict: bool = False,
    as_lazy: bool = False,
    seed: int = 273,
) -> Tuple[df_pl, ...] | Dict[str, df_pl]:
    """Split a dataset into non-overlapping len(rel_sizes) subsets."""
    rel_sizes_ = rel_sizes
    if isinstance(rel_sizes, tuple):
        rel_sizes_ = {f"subset_{i}": fraction for i, fraction in enumerate(rel_sizes)}

    df_lazy = df.lazy()

    idxs = pl.int_range(0, pl.count())
    if shuffle:
        idxs = idxs.shuffle(seed=seed)

    sizes = {subset: (rel_size * pl.count()).floor().cast(pl.Int64) for subset, rel_size in rel_sizes_.items()}

    if stratify_by:
        if any([dtype in (pl.Float64, pl.Float32) for dtype in df_lazy.select(stratify_by).dtypes]):
            raise NotImplementedError(
                "Stratification by a float column is not currently supported. "
                "Consider discretizing the column or using a different column."
            )
        if isinstance(stratify_by, str):
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
) -> Dict[str, pl.LazyFrame]:
    ...


@overload
def split_into_train_val_test(
    as_dict: Literal[True],
    as_lazy: Literal[False],
    *args,
    **kwargs,
) -> Dict[str, pl.DataFrame]:
    ...


@overload
def split_into_train_val_test(
    as_dict: Literal[False],
    as_lazy: Literal[False],
    *args,
    **kwargs,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    ...


@overload
def split_into_train_val_test(
    as_dict: Literal[False],
    as_lazy: Literal[True],
    *args,
    **kwargs,
) -> Tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    ...


@validate_rel_sizes
def split_into_train_val_test(
    df: df_pl,
    rel_sizes: Tuple[float, float, float],
    stratify_by: List[str] = None,
    shuffle: bool = True,
    as_dict: bool = False,
    as_lazy: bool = False,
    seed: int = 273,
) -> Tuple[df_pl, df_pl, df_pl] | Dict[str, df_pl]:
    if len(rel_sizes) != 3:
        raise ValueError("rel_sizes must be a tuple of length 3")

    subsets = split_into_subsets(df, rel_sizes, stratify_by, shuffle, as_dict=False, as_lazy=as_lazy, seed=seed)
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
) -> Dict[str, pl.LazyFrame]:
    ...


@overload
def split_into_train_eval(
    as_dict: Literal[True],
    as_lazy: Literal[False],
    *args,
    **kwargs,
) -> Dict[str, pl.DataFrame]:
    ...


@overload
def split_into_train_eval(
    as_dict: Literal[False],
    as_lazy: Literal[False],
    *args,
    **kwargs,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    ...


@overload
def split_into_train_eval(
    as_dict: Literal[False],
    as_lazy: Literal[True],
    *args,
    **kwargs,
) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    ...


def split_into_train_eval(
    df: df_pl,
    eval_fraction: float,
    stratify_by: List[str] = None,
    shuffle: bool = True,
    as_dict: bool = False,
    as_lazy: bool = False,
    seed: int = 273,
) -> Tuple[df_pl, df_pl] | Dict[str, df_pl]:
    if not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be between 0 and 1")

    rel_sizes = (1 - eval_fraction, eval_fraction)
    subsets = split_into_subsets(df, rel_sizes, stratify_by, shuffle, as_dict=False, as_lazy=as_lazy, seed=seed)
    if as_dict:
        return {k: v for k, v in zip(("train", "eval"), subsets)}
    else:
        return subsets


def split_into_k_folds_old(
    df: Union[pl.DataFrame, pl.LazyFrame],
    k: int,
    stratify_by: Union[str, List[str]] = None,
    shuffle: bool = True,
    as_dict: bool = True,
    as_lazy: bool = True,
    seed: int = 273,
) -> Dict[eval_set_name, Union[pl.DataFrame, pl.LazyFrame]]:
    df_lazy = df.lazy()

    fold_size = pl.count() // k
    idxs = pl.int_range(0, pl.count())

    if shuffle:
        idxs = idxs.shuffle(seed=seed)

    if stratify_by:
        fold_size = fold_size.over(stratify_by)
        idxs = idxs.over(stratify_by)

    folds = {i: {"train": None, "eval": None} for i in range(k)}
    for i in range(k):
        is_eval = (fold_size * i <= idxs) & (idxs < fold_size * (i + 1))

        if as_lazy:
            folds[i] = {"train": df_lazy.filter(~is_eval), "eval": df_lazy.filter(is_eval)}
        else:
            folds[i] = {"train": df_lazy.filter(~is_eval).collect(), "eval": df_lazy.filter(is_eval).collect()}

    return folds


def split_into_k_folds(
    df: Union[pl.DataFrame, pl.LazyFrame],
    k: int,
    stratify_by: Union[str, List[str]] = None,
    shuffle: bool = True,
    as_dict: bool = True,
    as_lazy: bool = False,
    seed: int = 273,
) -> Dict[eval_set_name, Union[pl.DataFrame, pl.LazyFrame]]:
    df_lazy = df.lazy()

    if stratify_by and any([dtype in (pl.Float64, pl.Float32) for dtype in df_lazy.select(stratify_by).dtypes]):
        raise NotImplementedError(
            "Stratification by a float column is not currently supported. "
            "Consider discretizing the column or using a different column."
        )

    folds = [{"train": None, "eval": None} for i in range(k)]
    for i in range(k):
        folds[i] = split_into_train_eval(
            df_lazy, 1 / k, stratify_by=stratify_by, shuffle=shuffle, as_dict=True, as_lazy=as_lazy, seed=seed
        )

    return folds


def k_fold(df: df_pl, k: int, shuffle: bool = True, seed: int = 273):
    return split_into_k_folds(df, k, stratify_by=None, shuffle=shuffle, as_dict=False, as_lazy=False, seed=seed)


get_k_folds = split_into_k_folds
stratified_k_fold = split_into_k_folds
train_test_split = split_into_train_eval
train_val_test_split = split_into_train_val_test
