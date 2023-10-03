from functools import wraps
from typing import Callable, Tuple

from loguru import logger
from polars import FLOAT_DTYPES, Int64, LazyFrame, count
from polars import selectors as cs

from polars_splitters.utils.wrapping_helpers import get_arg_value, replace_arg_value


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


def type_enforcer_train_eval(func: Callable) -> Callable:
    """Pre- and processing for the train_eval splitting functions, ensuring that the input is a LazyFrame and, if required, the output is a tuple of DataFrames."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Exception | None:
        df = get_arg_value(args, kwargs, "df_lazy", arg_index=0, expected_type=LazyFrame)
        args, kwargs = replace_arg_value(args, kwargs, "df_lazy", arg_index=0, new_value=df.lazy())

        df_train, df_eval = func(*args, **kwargs)

        as_lazy = get_arg_value(args, kwargs, "as_lazy", arg_index=5, default=False, expected_type=bool)

        if as_lazy:
            return (df_train, df_eval)
        else:
            return (df_train.collect(), df_eval.collect())

    return wrapper


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
