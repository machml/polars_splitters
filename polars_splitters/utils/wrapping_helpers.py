from polars_splitters.utils.type_enforcers import enforce_type


def get_arg_value(
    args,
    kwargs,
    arg_name,
    arg_index,
    default=None,
    expected_type=None,
    warn_on_recast=True,
):
    """Get the value of an argument from either args or kwargs."""
    arg_value = kwargs.get(arg_name, None)
    if arg_value is None:
        # args is always a tuple
        if len(args) >= arg_index + 1:
            arg_value = args[arg_index]
        elif default is not None:
            arg_value = default

    if expected_type:
        arg_value = enforce_type(arg_value, to_type=expected_type, warn=warn_on_recast)
    return arg_value


def replace_arg_value(args, kwargs, arg_name, arg_index, new_value):
    """Replace the value of an argument from either args or kwargs."""
    if arg_name in kwargs:
        kwargs[arg_name] = new_value
    else:
        args = list(args)
        args[arg_index] = new_value
        args = tuple(args)

    return args, kwargs
