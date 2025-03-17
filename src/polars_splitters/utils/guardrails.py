from collections.abc import Callable
from functools import wraps

from loguru import logger
from polars import Int64, LazyFrame
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


def get_lazyframe_size(df: LazyFrame) -> int:
    return df.select(pl_len()).collect().item()


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


def validate_splitting(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs) -> Exception | None:
        validate = get_arg_value(
            args,
            kwargs,
            "validate",
            arg_index=5,
            default=True,
            expected_type=bool,
        )
        if validate:
            # load arguments: args[0] stores theo func, the actual args start at index 1
            df = get_arg_value(args, kwargs, "df", arg_index=0, expected_type=LazyFrame)
            eval_rel_size = get_arg_value(
                args,
                kwargs,
                "eval_rel_size",
                arg_index=1,
                expected_type=float,
            )
            k = get_arg_value(args, kwargs, "k", arg_index=2, expected_type=int)
            stratify_by = get_arg_value(
                args,
                kwargs,
                "stratify_by",
                arg_index=3,
                expected_type=list,
                warn_on_recast=False,
            )

            # validate
            if k == 1 and eval_rel_size is None:
                raise ValueError(
                    f"Must specify either k>1 or eval_rel_size, got k={k} and eval_rel_size={eval_rel_size}.",
                )
            if k > 1 and eval_rel_size is not None:
                raise ValueError(
                    f"Cannot specify both k > 1 and eval_rel_size, got k={k} and eval_rel_size={eval_rel_size}.",
                )
            if k > 1:
                validate_var_within_bounds(k, (1, None))
                eval_rel_size_ = 1 / k
                if not isinstance(k, int):
                    raise TypeError(f"k must be of type int, got {type(k)}")

            elif eval_rel_size is not None:
                validate_var_within_bounds(eval_rel_size, (0.0, 1.0))
                if not isinstance(eval_rel_size, float):
                    raise TypeError(
                        f"eval_rel_size must be of type float, got {type(eval_rel_size)}",
                    )

                eval_rel_size_ = eval_rel_size

            input_size = get_lazyframe_size(df)

            if stratify_by:
                # validate stratification feasibility (size_input, eval_rel_size (or k), n_strata, stratify_by)
                n_strata = df.select(stratify_by).collect().n_unique()

                eval_size_targeted = (
                    df.select(
                        (eval_rel_size_ * pl_len()).round(0).clip(lower_bound=1).cast(Int64),
                    )
                    .collect()
                    .item()
                )

                if eval_rel_size_ <= 0.5:
                    smallest_set, smallest_set_size = ("eval", eval_size_targeted)
                else:
                    smallest_set, smallest_set_size = (
                        "train",
                        input_size - eval_size_targeted,
                    )

                if smallest_set_size < n_strata:
                    f"""
                    Unable to generate the data splits for the data df and the configuration attempted for {_get_eval_sizing_measure(k)} and stratify_by.
                    For the stratification to work, the size of the smallest set (currently {smallest_set}: {smallest_set_size})
                    must be at least as large as the number of strata (currently {n_strata}), i.e. the number of unique row-wise
                    combinations of values in the stratify_by columns (currently {stratify_by}).

                    {_get_suggestion_for_loosening_stratification(k)}
                    """

        folds = func(*args, **kwargs)

        if validate:
            rel_size_deviation_tolerance = get_arg_value(
                args,
                kwargs,
                "rel_size_deviation_tolerance",
                arg_index=9,
                default=0.1,
                expected_type=float,
            )

            for i, fold in enumerate(folds):
                df_eval = fold["eval"]

                eval_rel_size_actual = get_lazyframe_size(df_eval) / input_size

                rel_size_deviation = abs(eval_rel_size_actual - eval_rel_size_)

                logger.info(
                    f"fold {i + 1}/{k}, k: {k}, eval_rel_size: {eval_rel_size_}, eval_rel_size_actual: {eval_rel_size_actual}, rel_size_deviation_tolerance: {rel_size_deviation_tolerance}, rel_size_deviation: {rel_size_deviation}",
                )

                if rel_size_deviation > rel_size_deviation_tolerance + 1e-6:
                    raise ValueError(
                        f"""
                            The actual relative size of the eval set ({eval_rel_size_actual}) deviates from the requested relative size ({eval_rel_size_})
                            by more than the specified tolerance ({rel_size_deviation_tolerance}).

                            {_get_suggestion_for_loosening_stratification(k)}
                            """,
                    )

        return folds

    return wrapper


def enforce_input_outputs_expected_types(func: Callable) -> Callable:
    """Pre- and processing for the train_eval splitting functions, ensuring that the input is a LazyFrame and, if required, the output is a tuple of DataFrames."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Exception | None:
        df = get_arg_value(args, kwargs, "df", arg_index=0, expected_type=LazyFrame)
        args, kwargs = replace_arg_value(
            args,
            kwargs,
            "df",
            arg_index=0,
            new_value=df.lazy(),
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
            folds = [{subset_name: df.collect() for subset_name, df in fold.items()} for fold in folds]
        if not as_dict:
            folds = [tuple(fold.values()) for fold in folds]
        if k == 1:
            folds = folds[0]

        return folds

    return wrapper
