import math
from typing import Dict, List, Literal, Tuple, Union, overload

import polars as pl

df_pl = pl.DataFrame | pl.LazyFrame
eval_set_name = Literal["val", "test"]


@overload
def _split(
    k: Literal[None],
    as_dict: Literal[False],
    as_lazy: Literal[True],
    *args,
    **kwargs,
) -> Tuple[pl.LazyFrame]:
    ...


@overload
def _split(
    k: Literal[None],
    as_dict: Literal[False],
    as_lazy: Literal[False],
    *args,
    **kwargs,
) -> Tuple[pl.DataFrame]:
    ...


@overload
def _split(
    eval_frac: Literal[None],
    fracs: Literal[None],
    as_dict: Literal[False],
    as_lazy: Literal[True],
    *args,
    **kwargs,
) -> List[Tuple[pl.LazyFrame]]:
    # k-fold use cases
    ...


@overload
def _split(
    eval_frac: Literal[None],
    fracs: Literal[None],
    as_dict: Literal[True],
    as_lazy: Literal[True],
    *args,
    **kwargs,
) -> List[Dict[eval_set_name, pl.LazyFrame]]:
    # k-fold use cases
    ...


@overload
def _split(
    eval_frac: Literal[None],
    fracs: Literal[None],
    as_dict: Literal[False],
    as_lazy: Literal[False],
    *args,
    **kwargs,
) -> List[Tuple[pl.DataFrame]]:
    # k-fold use cases
    ...


@overload
def _split(
    eval_frac: Literal[None],
    fracs: Literal[None],
    as_dict: Literal[True],
    as_lazy: Literal[False],
    *args,
    **kwargs,
) -> List[Dict[eval_set_name, pl.DataFrame]]:
    # k-fold use cases
    ...


def _split(
    df: df_pl,
    eval_frac: float,
    fracs: Tuple[float, float, float],
    k: int,
    stratify_by: List[str] = None,
    shuffle: bool = True,
    as_dict: bool = True,
    as_lazy: bool = False,
    seed: int = 25,
) -> Tuple[df_pl] | List[Tuple[df_pl]] | List[Dict[eval_set_name, df_pl]]:
    """Split Polar dataframes into subparts.
    A generic split function that abstracts split_into_train_eval, split_into_train_val_test, and split_into_k_folds.
    """
    df_lazy = df.lazy()

    idxs = pl.int_range(0, pl.count())
    if shuffle:
        idxs = idxs.shuffle(seed=seed)

    if k:  # k-fold split
        size = {"eval": pl.count() // k}
    elif eval_frac:  # train/eval split
        size = {"eval": (eval_frac * pl.count()).floor().cast(pl.Int64)}
    elif fracs:  # train/val/test split
        size = {
            "val": (fracs[-1] * pl.count()).floor().cast(pl.Int64),
            "test": (fracs[1] / (fracs[0] + fracs[1]) * pl.count()).floor().cast(pl.Int64),
        }
    else:
        raise ValueError("Must specify either k, eval_frac, or fracs")

    if stratify_by:
        idxs = idxs.over(stratify_by)
        size = {k: v.over(stratify_by) for k, v in size.items()}

    if not k:
        k = 1

    folds = [{"train": None} for i in range(k)]
    for fold in folds:
        for j, subset in enumerate(size.keys()):
            is_subset = (size[subset] * j <= idxs) & (idxs < size[subset] * (j + 1))
            fold[subset] = df_lazy.filter(is_subset) if as_lazy else df_lazy.filter(is_subset).collect()
            fold["train"] = df_lazy.filter(~is_subset) if as_lazy else df_lazy.filter(~is_subset).collect()

    if not as_lazy:
        ....collect()

    return ...


def split_into_train_eval(
    df: df_pl,
    eval_frac: float,
    stratify_by: List[str] = None,
    shuffle: bool = True,
    as_dict: bool = True,
    as_lazy: bool = False,
    seed: int = 25,
) -> Tuple[df_pl, df_pl]:
    fracs = (1 - eval_frac, 0, eval_frac)
    df_train, _, df_eval = split_into_train_val_test(df, fracs, stratify_by, shuffle, as_dict, as_lazy, seed)

    return df_train, df_eval


@overload
def split_into_train_val_test(
    as_dict: Literal[False],
    as_lazy: Literal[True],
    *args,
    **kwargs,
) -> Tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    ...


def split_into_train_val_test(
    df: df_pl,
    fracs: Tuple[float, float, float],
    stratify_by: List[str] = None,
    shuffle: bool = True,
    as_dict: bool = True,
    as_lazy: bool = False,
    seed: int = 25,
) -> Tuple[df_pl, df_pl, df_pl]:
    if not math.isclose(sum(fracs), 1.0, abs_tol=1e-6):
        raise ValueError(f"Sum of fracs must be 1, but got {sum(fracs)}")

    df_lazy = df.lazy()

    idxs = pl.int_range(0, pl.count())
    if shuffle:
        idxs = idxs.shuffle(seed=seed)

    size = {
        "train": (fracs[0] * pl.count()).floor().cast(pl.Int64),
        "val": (fracs[1] * pl.count()).floor().cast(pl.Int64),
        "test": (fracs[2] * pl.count()).floor().cast(pl.Int64),
    }

    if stratify_by:
        idxs = idxs.over(stratify_by)
        size = {k: v.over(stratify_by) for k, v in size.items()}

    is_train = idxs <= size["train"]
    is_val = (size["train"] < idxs) & (idxs <= size["train"] + size["val"])
    is_test = (size["train"] + size["val"] < idxs) & (idxs <= size["train"] + size["val"] + size["test"])

    subsets = {"train": df_lazy.filter(is_train), "test": df_lazy.filter(is_test)}
    if fracs[1] == 0:
        subsets["val"] = None
    else:
        subsets["val"] = df_lazy.filter(is_val)

    if not as_lazy:
        subsets = {k: v.collect() for k, v in subsets.items()}

    if not as_dict:
        return subsets["train"], subsets["val"], subsets["test"]
    else:
        return subsets


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
        rel_sizes_ = {f"subset_{i}": rel_size for i, rel_size in enumerate(rel_sizes)}

    df_lazy = df.lazy()

    idxs = pl.int_range(0, pl.count())
    if shuffle:
        idxs = idxs.shuffle(seed=seed)

    sizes = {subset: (rel_size * pl.count()).floor().cast(pl.Int64) for subset, rel_size in rel_sizes.items()}

    if stratify_by:
        idxs = idxs.over(stratify_by)
        sizes = {k: v.over(stratify_by) for k, v in sizes.items()}

    subsets = {subset: None for subset in sizes.keys()}
    for i, subset in enumerate(subsets.keys()):
        sizes_already_used = sum(list(sizes.values())[:i])
        is_subset = (sizes_already_used <= idxs) & (idxs < sizes_already_used + sizes[subset])
        subsets[subset] = df_lazy.filter(is_subset)

    if not as_lazy:
        subsets = {k: v.collect() for k, v in subsets.items()}

    if not as_dict:
        return tuple(subsets.values())
    else:
        return subsets


def split_into_k_folds(
    df: Union[pl.DataFrame, pl.LazyFrame],
    k: int,
    stratify_by: Union[str, List[str]] = None,
    shuffle: bool = True,
    as_dict: bool = True,
    as_lazy: bool = True,
    seed: int = 25,
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


get_k_folds = split_into_k_folds
train_test_split = split_into_train_eval
