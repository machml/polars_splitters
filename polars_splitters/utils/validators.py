import math
from functools import wraps
from typing import Any, Callable, Tuple

from loguru import logger
from polars import FLOAT_DTYPES, Int64, LazyFrame, count
from polars import selectors as cs

from polars_splitters.utils.wrapping_helpers import get_arg_value


def _get_suggestion_for_loosening_stratification(func_name: str) -> str:
    if func_name == "split_into_train_eval":
        balance_suggestion = "using an eval_rel_size closer to 0.5"
    elif func_name == "split_into_k_folds":
        balance_suggestion = "using a smaller k"

    return f"""
            Consider:
            (a) {balance_suggestion},
            (b) using fewer columns in stratify_by columns,
            (c) disabling stratification altogether (stratify_by=None),
            (d) using a larger input dataset df or
            (e) increasing the tolerance for the relative size deviation (rel_size_deviation_tolerance).
    """


def get_lazyframe_size(df: LazyFrame) -> int:
    return df.select(count()).collect().item()


def validate_var_within_bounds(
    var: float, bounds: Tuple[float | None, float | None] | Tuple[int | None, int | None]
) -> Exception | None:
    """Ensure that the variable is within the specified bounds."""

    if bounds[0] and bounds[1]:
        if not bounds[0] < var < bounds[1]:
            raise ValueError(f"var must be between {bounds[0]} and {bounds[1]}, got {var}")
    elif bounds[0] and not bounds[1]:
        if not bounds[0] < var:
            raise ValueError(f"var must be greater than {bounds[0]}, got {var}")
    elif not bounds[0] and bounds[1]:
        if not var < bounds[1]:
            raise ValueError(f"var must be less than {bounds[1]}, got {var}")


def validate_splitting_train_eval(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Exception | None:
        validate = get_arg_value(args, kwargs, "validate", arg_index=5, default=True, expected_type=bool)
        if validate:
            # load arguments: args[0] stores theo func, the actual args start at index 1
            df_lazy = get_arg_value(args, kwargs, "df_lazy", arg_index=0, expected_type=LazyFrame)
            eval_rel_size = get_arg_value(args, kwargs, "eval_rel_size", arg_index=1, expected_type=float)
            stratify_by = get_arg_value(
                args, kwargs, "stratify_by", arg_index=2, expected_type=list, warn_on_recast=False
            )

            # validate
            validate_var_within_bounds(eval_rel_size, (0.0, 1.0))

            if stratify_by:
                # validate stratification dtypes
                stratification_columns_of_float_type = (
                    df_lazy.select(stratify_by).select(cs.by_dtype(FLOAT_DTYPES)).schema
                )
                if stratification_columns_of_float_type:
                    raise NotImplementedError(
                        f"""
                            Attempted to stratify based on float column(s): {stratification_columns_of_float_type}.
                            This is not currently supported. Consider discretizing the column first or using a different column.
                            """
                    )

                # validate stratification feasibility (size_input, eval_rel_size, n_strata, stratify_by)
                n_strata = df_lazy.select(stratify_by).collect().n_unique()
                input_size = get_lazyframe_size(df_lazy)
                eval_size_targeted = (
                    df_lazy.select((eval_rel_size * count()).round(0).clip_min(1).cast(Int64)).collect().item()
                )

                if eval_rel_size <= 0.5:
                    smallest_set, smallest_set_size = ("eval", eval_size_targeted)
                else:
                    smallest_set, smallest_set_size = ("train", input_size - eval_size_targeted)

                logger.debug(
                    f"""input_size: {input_size}, eval_size_targeted: {eval_size_targeted}, train_size_targeted: {input_size - eval_size_targeted}
                        smallest_set: {smallest_set}, smallest_set_size: {smallest_set_size}, n_strata: {n_strata}, stratify_by: {stratify_by}
                        """
                )

                if smallest_set_size < n_strata:
                    f"""
                    Unable to generate the data splits for the data df and the configuration attempted for eval_rel_size and stratify_by.
                    For the stratification to work, the size of the smallest set (currently {smallest_set}: {smallest_set_size})
                    must be at least as large as the number of strata (currently {n_strata}), i.e. the number of unique row-wise
                    combinations of values in the stratify_by columns (currently {stratify_by}).

                    {_get_suggestion_for_loosening_stratification(func.__name__)}
                    """

        df_train, df_eval = func(*args, **kwargs)

        if validate:
            rel_size_deviation_tolerance = get_arg_value(
                args, kwargs, "rel_size_deviation_tolerance", arg_index=6, default=0.1, expected_type=float
            )
            eval_rel_size_actual = get_lazyframe_size(df_eval) / input_size

            rel_size_deviation = abs(eval_rel_size_actual - eval_rel_size)

            logger.debug(
                f"eval_rel_size: {eval_rel_size}, eval_rel_size_actual: {eval_rel_size_actual}, rel_size_deviation_tolerance: {rel_size_deviation_tolerance}, rel_size_deviation: {rel_size_deviation}"
            )

            if rel_size_deviation > rel_size_deviation_tolerance + 1e-6:
                raise ValueError(
                    f"""
                        The actual relative size of the eval set ({eval_rel_size_actual}) deviates from the requested relative size ({eval_rel_size})
                        by more than the specified tolerance ({rel_size_deviation_tolerance}).

                        {_get_suggestion_for_loosening_stratification(func.__name__)}
                        """
                )

        return (df_train, df_eval)

    return wrapper


from polars import Float32, Float64, Int64, LazyFrame, count


def validate_k(k: int):
    if not isinstance(k, int):
        raise TypeError(f"k must be of type int, got {type(k)}")

    if k < 2:
        raise ValueError(f"k must be greater than 1, got {k}")


def validate_eval_rel_size(eval_rel_size: float):
    if not isinstance(eval_rel_size, float):
        raise TypeError(f"eval_rel_size must be of type float, got {type(eval_rel_size)}")

    if not 0 < eval_rel_size < 1:
        raise ValueError(f"eval_rel_size must be between 0 and 1, got {eval_rel_size}")


def validate_rel_sizes(rel_sizes_: Tuple[float, ...] | Dict[str, float], rel_sizes_length: int):
    """Assumes that rel_sizes is the second argument of the function being wrapped."""

    if isinstance(rel_sizes_, dict):
        rel_sizes_ = tuple(rel_sizes_.values())

    if len(rel_sizes_) != rel_sizes_length:
        raise ValueError(f"rel_sizes must have length {rel_sizes_length}")

    # sum should be 1
    if not math.isclose(sum(rel_sizes_), 1.0, abs_tol=1e-6):
        raise ValueError("rel_sizes must sum to 1")

    # all values should be non-negative
    if any(rel_size < 0 for rel_size in rel_sizes_):
        raise ValueError("rel_sizes must be non-negative")

    # all values should be less than 1
    if any(rel_size > 1 for rel_size in rel_sizes_):
        raise ValueError("rel_sizes must be less than 1")

    isnt_tuple_nor_dict = not (isinstance(rel_sizes_, tuple) or isinstance(rel_sizes_, dict))
    arent_tuple_values_all_floats = isinstance(rel_sizes_, tuple) and not all(
        isinstance(rel_size, float) for rel_size in rel_sizes_
    )
    arent_dict_values_all_floats = isinstance(rel_sizes_, dict) and not all(
        isinstance(rel_size, float) for rel_size in rel_sizes_.values()
    )

    if isnt_tuple_nor_dict or arent_tuple_values_all_floats or arent_dict_values_all_floats:
        raise TypeError("rel_sizes must be of type Tuple[float], Dict[Any, float]")


def validate_stratification_dtypes(df_: LazyFrame, stratify_by_: list[str]):
    """Ensure that none of the columns specified in stratify_by are of float type (i.e. non-discrete valued)."""
    dtypes_stratify_by = df_.select(stratify_by_).dtypes

    if any([dtype in (Float64, Float32) for dtype in dtypes_stratify_by]):
        raise NotImplementedError(
            "Stratification by a float column is not currently supported. "
            "Consider discretizing the column or using a different column."
        )


def validate_stratification_kfolds(df_: LazyFrame, k_, stratify_by_: list[str]):
    validate_stratification_dtypes(df_, stratify_by_)

    n = df_.select(count()).collect().item()
    eval_size = n // k_
    n_unique_combinations = df_.select(stratify_by_).n_unique()
    if eval_size < n_unique_combinations:
        f"""
        Unable to generate the k folds for the data df and the configuration you set for rel_sizes and stratify_by.
        For the stratification to work, the size of the evaluation set in each fold `len(df) // k` (currently {eval_size})
        must be at least as large as the number of unique combinations of values found for the columns in stratify_by={stratify_by_} (currently {n_unique_combinations}).

        Consider:
        (a) using a smaller number of folds k,
        (b) using fewer columns in stratify_by columns,
        (c) disabling stratification altogether (stratify_by=None) or
        (d) using a larger input dataset df.
        """


def validate_stratification_train_eval(df_: LazyFrame, eval_rel_size_: float, stratify_by_: list[str]):
    validate_stratification_dtypes(df_, stratify_by_)

    n = df_.select(count()).collect().item()
    eval_size = (eval_rel_size_ * count()).floor().cast(Int64).item()

    is_train_size_smaller_than_eval_size = eval_rel_size_ > 0.5

    if not is_train_size_smaller_than_eval_size:
        smallest_set, smallest_set_size = ("eval", eval_size)
    else:
        smallest_set, smallest_set_size = ("train", n - eval_size)

    n_unique_combinations = df_.select(stratify_by_).n_unique()

    if smallest_set < n_unique_combinations:
        f"""
        Unable to generate the data splits for the data df and the configuration attempted for eval_rel_size and stratify_by.
        For the stratification to work, the size of the smallest set (currently {smallest_set}: {smallest_set_size})
        must be at least as large as the number of unique combinations of values found for the columns in stratify_by={stratify_by_} (currently {n_unique_combinations}).

        Consider:
        (a) using a larger rel_size {smallest_set} (via eval_rel_size)),
        (b) using fewer columns in stratify_by columns,
        (c) disabling stratification altogether (stratify_by=None) or
        (d) using a larger input dataset df.
        """


def validate_stratification_train_val_test(df_: LazyFrame, rel_sizes_: Tuple[float, ...], stratify_by_: list[str]):
    validate_stratification_dtypes(df_, stratify_by_)

    n = df_.select(count()).collect().item()

    if isinstance(rel_sizes_, tuple):
        rel_sizes_ = dict(zip(["train", "val", "test"], rel_sizes_))

    sizes = {k: (v * n).floor().cast(Int64).item() for k, v in rel_sizes_.items()}

    smallest_set, smallest_set_size = min(sizes.items())
    n_unique_combinations = df_.select(stratify_by_).n_unique()

    if smallest_set_size < n_unique_combinations:
        f"""
        Unable to generate the data splits for the data df and the configuration attempted for rel_sizes and stratify_by.
        For the stratification to work, the size of the smallest set (currently {smallest_set}: {smallest_set_size})
        must be at least as large as the number of unique combinations of values found for the columns in stratify_by={stratify_by_} (currently {n_unique_combinations}).

        Consider:
        (a) using a larger rel_size for {smallest_set},
        (b) using fewer columns in stratify_by columns,
        (c) disabling stratification altogether (stratify_by=None) or
        (d) using a larger input dataset df.
        """


def validate_inputs(func: Callable) -> Any:
    """Validate (a) rel_sizes, as well as (b) the combination of df, rel_sizes (or k), and stratify_by.

    Assumes the inputs are in one of the following orders:
    - df, rel_sizes, stratify_by, ...
    - df, rel_sizes, stratify_by, ...
    - df, k, stratify_by, ...
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        df_ = kwargs.get("df", args[0])
        df_ = df_.lazy()  # ensure df is a LazyFrame

        stratify_by_ = kwargs.get("stratify_by", None)
        if stratify_by_ is None:
            stratify_by_ = args[2] if len(args) >= 3 else None

        if stratify_by_ and not isinstance(stratify_by_, list):
            stratify_by_ = [stratify_by_]

        if func.__name__ == "split_into_k_folds":
            k_ = kwargs.get("k", args[1])
            validate_k(k_)
            if stratify_by_:
                validate_stratification_kfolds(df_, k_, stratify_by_)

        elif func.__name__ == "split_into_train_test":
            eval_rel_size_ = kwargs.get("eval_rel_size", args[1])
            validate_eval_rel_size(eval_rel_size_)
            if stratify_by_:
                validate_stratification_train_eval(df_, eval_rel_size_, stratify_by_)

        elif func.__name__ == "split_into_train_val_test":
            rel_sizes_ = kwargs.get("rel_sizes", args[1])
            validate_rel_sizes(rel_sizes_, rel_sizes_length=3)
            if stratify_by_:
                validate_stratification_train_val_test(df_, rel_sizes_, stratify_by_)

        return func(*args, **kwargs)

    return wrapper
