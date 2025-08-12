from collections.abc import Callable
from functools import wraps

from polars import DataFrame, Int64
from polars import len as pl_len

from polars_splitters.utils.wrapping_helpers import get_arg_value, replace_arg_value


def _get_suggestion_for_loosening_stratification(k: int) -> str:
    if k > 1:
        balance_suggestion = "using a smaller k"
    elif k == 1:
        balance_suggestion = "using an eval_rel_size closer to 0.5"

    return f"""
            Consider:
            (a) {balance_suggestion},
            (b) using fewer columns in stratify_by columns,
            (c) disabling stratification altogether (stratify_by=None),
            (d) using a larger input dataset df or
            (e) increasing the tolerance for the relative size deviation (rel_size_deviation_tolerance).
    """


def _get_eval_sizing_measure(k: int) -> str:
    if k > 1:
        return "k"
    if k == 1:
        return "eval_rel_size"
    else:
        error_message = f"k should be an integer k>=1, got {k} (dtype: {type(k)})"
        raise ValueError(error_message)


def validate_var_within_bounds(
    var: float,
    bounds: tuple[float | int | None, float | int | None],
) -> Exception | None:
    """Ensure that the variable is within the specified bounds."""
    if bounds[0] and bounds[1]:
        if not bounds[0] < var < bounds[1]:
            raise ValueError(
                f"var must be between {bounds[0]} and {bounds[1]}, got {var}",
            )
    elif bounds[0] and not bounds[1]:
        if not bounds[0] < var:
            raise ValueError(f"var must be greater than {bounds[0]}, got {var}")
    elif not bounds[0] and bounds[1]:
        if not var < bounds[1]:
            raise ValueError(f"var must be less than {bounds[1]}, got {var}")


def validate_splitting(
    folds: list[dict[str, DataFrame]],
    df: DataFrame,
    eval_rel_size: float,
    k: int,
    stratify_by: list[str] | None,
    rel_size_deviation_tolerance: float,
) -> None:
    """
    Validate the output folds for train/eval splitting.
    Checks that the actual relative size of the eval set is within tolerance of the requested size,
    and that stratification is feasible if requested.
    """
    if k > 1:
        eval_rel_size_ = 1 / k
    else:
        eval_rel_size_ = eval_rel_size

    input_height = df.height

    if stratify_by:
        # validate stratification feasibility (size_input, eval_rel_size (or k), n_strata, stratify_by)
        n_strata = df.select(stratify_by).n_unique()
        eval_size_targeted = df.select(
            (eval_rel_size_ * pl_len()).round(0).clip(lower_bound=1).cast(Int64),
        ).item()
        if eval_rel_size_ <= 0.5:
            smallest_set_size = eval_size_targeted
        else:
            smallest_set_size = input_height - eval_size_targeted
        if smallest_set_size < n_strata:
            raise ValueError(
                f"""
                Unable to generate the data splits for the data df and the configuration attempted for {_get_eval_sizing_measure(k)} and stratify_by.
                For the stratification to work, the size of the smallest set (currently {smallest_set_size})
                must be at least as large as the number of strata (currently {n_strata}), i.e. the number of unique row-wise
                combinations of values in the stratify_by columns (currently {stratify_by}).

                {_get_suggestion_for_loosening_stratification(k)}
                """,
            )

    # Validate output folds
    for fold in folds:
        df_eval = fold["eval"]
        eval_rel_size_actual = df_eval.height / input_height
        rel_size_deviation = abs(eval_rel_size_actual - eval_rel_size_)
        if rel_size_deviation > rel_size_deviation_tolerance + 1e-6:
            raise ValueError(
                f"""
                The actual relative size of the eval set ({eval_rel_size_actual}) deviates from the requested relative size ({eval_rel_size_})
                by more than the specified tolerance ({rel_size_deviation_tolerance}).

                {_get_suggestion_for_loosening_stratification(k)}
                """,
            )
    # No return value; raises on error


def enforce_input_outputs_expected_types(func: Callable) -> Callable:
    """Pre- and processing for the train_eval splitting functions, ensuring that the input is a LazyFrame and, if required, the output is a tuple of DataFrames."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Exception | None:
        df = get_arg_value(args, kwargs, "df", arg_index=0, expected_type=DataFrame)

        assert isinstance(df, DataFrame), "df must be a pl.DataFrame"
        args, kwargs = replace_arg_value(
            args,
            kwargs,
            "df",
            arg_index=0,
            new_value=df,
        )

        k = get_arg_value(args, kwargs, "k", arg_index=2, expected_type=int)

        folds = func(*args, **kwargs)

        as_lazy = get_arg_value(
            args,
            kwargs,
            "as_lazy",
            arg_index=6,
            default=False,
            expected_type=bool,
        )
        as_dict = get_arg_value(
            args,
            kwargs,
            "as_dict",
            arg_index=7,
            default=False,
            expected_type=bool,
        )

        if not as_lazy:
            folds = [{subset_name: df for subset_name, df in fold.items()} for fold in folds]
        if not as_dict:
            folds = [tuple(fold.values()) for fold in folds]
        if k == 1:
            folds = folds[0]

        return folds

    return wrapper
    return wrapper
