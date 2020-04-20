from functools import wraps, partialmethod, partial


def ensure_valid_image(num_dims_total=None, num_dims_spatial=None, num_channels=None,
                       keep_num_dims=True):
    """

    Parameters
    ----------
    num_dims_total: tuple of ints or None
        If not None, checks whether the number of the input dimensions is among
        the specified values.
    num_dims_spatial: tuple of ints or None
        If not None, checks whether the number of the spatial (i.e. total - 1)
        dimensions is among the specified values.
    num_channels: tuple of ints or None
        If not None, checks whether the shape of the last dimension is among
        the specified values.
    keep_num_dims: bool
        If True, adds the trailing dimensions to the result to match the input.

    Returns
    -------

    Raises
    ------
    ValueError:
        If one or several of the checks failed.
    """
    def inner_decorator(method):
        @wraps(method)
        def wrapped(*args, **kwargs):
            if "img" in kwargs:
                img = kwargs["img"]
            else:
                img = args[1]  # 0th arg is `self`
            shape_in, ndim_in = img.shape, img.ndim

            # Ensure the input conforms with the num_dims, shape, etc requirements
            if num_dims_total is not None:
                if ndim_in not in num_dims_total:
                    msg = f"Unsupported number of dimensions {ndim_in}"
                    raise ValueError(msg)

            if num_dims_spatial is not None:
                if (ndim_in - 1) not in num_dims_spatial:
                    msg = f"Unsupported number of spatial dimensions {ndim_in - 1}"
                    raise ValueError(msg)

            if num_channels is not None:
                if shape_in[-1] not in num_channels:
                    msg = f"Unsupported number of channels {shape_in[-1]}"
                    raise ValueError(msg)

            # Execute the wrapped function
            result = method(*args, **kwargs)

            # Implement the required consistency
            shape_out, ndim_out = result.shape, result.ndim

            if keep_num_dims and ndim_out < ndim_in:
                # Add trailing dimensions to match the input
                sel = (..., ) + (None, ) * (ndim_in - ndim_out)
                result = result[sel]

            return result
        return wrapped
    return inner_decorator


def validate_parameter(parameter, allowed_modes, default_value, basic_type=str, heritable=True):
    """
    Validates the parameter and wraps it into a tuple with the
    inheritance option (if parameter is not a tuple already).
    In this case the parameter will become a tuple (parameter, 'inherit'),
    which will indicate that the stream settings will override this parameter.
    In case if the parameter is already a tuple specified as parameter=(value, 'strict'), then the parameter
    will not be overrided.

    Parameters
    ----------
    parameter : object
        The value of the parameter
    allowed_modes : dict or set
        Allowed values for the parameter
    default_value : object
        Default value to substitute if the parameter is None
    basic_type : type
        Type of the parameter.
    heritable : bool
        Whether to check for heretability option.

    Returns
    -------
    out : tuple
        New parameter value wrapped into a tuple.

    """

    if parameter is None:
        parameter = default_value

    if isinstance(parameter, basic_type) and heritable:
        parameter = (parameter, "inherit")

    if isinstance(parameter, list):
        parameter = tuple(parameter)

    if isinstance(parameter, tuple) and heritable:
        if len(parameter) != 2:
            raise ValueError
        if not isinstance(parameter[0], basic_type):
            raise TypeError
        if parameter[0] not in allowed_modes:
            raise ValueError
        if parameter[1] not in {"inherit", "strict"}:
            raise ValueError
    elif heritable:
        raise NotImplementedError

    return parameter


def validate_numeric_range_parameter(parameter, default_val, min_val=None, max_val=None):
    """Validates the range-type parameter, e.g. angle in Random Rotation.

    Parameters
    ----------
    parameter : tuple or None
        The value of the parameter
    default_val : object
        Default value of the parameter if it is None.
    min_val: None or float or int
        Check whether the parameter is greater or equal than this. Optional.
    max_val: None or float or int
        Check whether the parameter is less or equal than this. Optional.
    Returns
    -------
    out : tuple
        Parameter value, passed all the checks.

    """

    if not isinstance(default_val, tuple):
        raise TypeError

    if parameter is None:
        parameter = default_val

    if isinstance(parameter, list):
        parameter = tuple(parameter)

    if not isinstance(parameter, tuple):
        raise TypeError

    if len(parameter) != 2:
        raise ValueError

    if parameter[0] > parameter[1]:
        raise ValueError

    if not (isinstance(parameter[0], (int, float)) and isinstance(parameter[1], (int, float))):
        raise TypeError("Incorrect type of the parameter!")

    if min_val is not None:
        if parameter[0] < min_val or parameter[1] < min_val:
            raise ValueError

    if max_val is not None:
        if parameter[0] > max_val or parameter[1] > max_val:
            raise ValueError

    return parameter
