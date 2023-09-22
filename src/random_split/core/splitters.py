import math
from typing import Dict, List, Tuple, Union

import polars as pl

# pl.set_random_seed(25)  # TODO: remove this


def split_train_eval(
    df: Union[pl.DataFrame, pl.LazyFrame],
    eval_frac: float,
    stratify_by: List[str] = None,
    lazy: bool = False,
    seed: int = 25,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    shuffled_idxs = pl.int_range(0, pl.count()).shuffle(seed=seed)
    idx_start_eval = (eval_frac * pl.count()).floor().cast(pl.Int32)

    if stratify_by:
        shuffled_idxs = shuffled_idxs.over(stratify_by)
        idx_start_eval = idx_start_eval.over(stratify_by)

    is_eval = shuffled_idxs < idx_start_eval

    df_train, df_eval = df.filter(~is_eval), df.filter(is_eval)

    if isinstance(df, pl.LazyFrame) and not lazy:
        return df_train.collect(), df_eval.collect()
    else:
        return df_train, df_eval


def split_train_val_test(
    df: Union[pl.DataFrame, pl.LazyFrame],
    fracs: Tuple[float, float, float],
    stratify_by: List[str] = None,
    lazy: bool = False,
    seed: int = 25,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if not math.isclose(sum(fracs), 1.0, abs_tol=1e-6):
        raise ValueError(f"Sum of fracs must be 1, but got {sum(fracs)}")

    test_frac = fracs[-1]
    df_train_e_val, df_test = split_train_eval(df, test_frac, stratify_by, lazy=True, seed=seed)

    val_frac = fracs[1] / (fracs[0] + fracs[1])
    df_train, df_val = split_train_eval(df_train_e_val, val_frac, stratify_by, lazy=True, seed=seed)

    if isinstance(df, pl.LazyFrame) and not lazy:
        return df_train.collect(), df_val.collect(), df_test.collect()
    else:
        return df_train, df_val, df_test


def get_k_folds(
    df: Union[pl.DataFrame, pl.LazyFrame],
    k: int,
    stratify_by: Union[str, List[str]] = None,
    lazy: bool = True,
    shuffle: bool = True,
    seed: int = 25,
) -> Dict[str, Union[pl.DataFrame, pl.LazyFrame]]:
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

        if lazy:
            folds[i] = {"train": df_lazy.filter(~is_eval), "eval": df_lazy.filter(is_eval)}
        else:
            folds[i] = {"train": df_lazy.filter(~is_eval).collect(), "eval": df_lazy.filter(is_eval).collect()}

    return folds
