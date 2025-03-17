from typing import Any

from loguru import logger
from polars import DataFrame, LazyFrame

__all__ = ["enforce_type", "ensure_type"]


def ensure_type(var: Any, to_type: type, warn: bool = False) -> Any:
    """Ensure that the variable is of the specified type if it exists."""
    from_type = type(var)

    if var is None or from_type == to_type:
        return var

    if warn:
        logger.info(f"Converting from {from_type} to {to_type}.")

    if (from_type, to_type) == (LazyFrame, DataFrame):
        return var.collect()
    if (from_type, to_type) == (DataFrame, LazyFrame):
        return var.lazy()
    if (from_type, to_type) == (dict, tuple):
        return tuple(var.values())
    if to_type == list:
        return [var]
    raise NotImplementedError(f"Cannot enforce a {from_type} into a {to_type}.")


enforce_type = ensure_type
