from functools import wraps


class Serializable(object):
    registry = {}

    def serialize(self, fmt="dict"):
        """Method returns an ordered dict, describing the object.

        Parameters
        ----------
        fmt : str
            Serialization type. Can be dict, json or yaml.
        Returns
        -------
        out : OrderedDict
            OrderedDict, ready for json serialization.

        """
        if fmt not in ["dict", "json", "yaml"]:
            raise ValueError("Format of serialization can only be dict, json or yaml!")
        d = {}
        for item in self.__dict__.items():
            # state_dict is not serialized
            if item[0] == "state_dict":
                continue

            if hasattr(item[1], "serialize"):
                d[item[0]] = item[1].serialize()
            elif item[0] != "transforms":
                d[item[0]] = item[1]
            elif isinstance(item[1], (tuple, list)) and item[0] == "transforms":
                d[item[0]] = [{x.__class__.__name__: x.serialize()} for x in item[1]]

        return d

    def __init_subclass__(cls, **kwargs):
        super(Serializable, cls).__init_subclass__(**kwargs)
        cls.registry[f"{cls.__module__}.{cls.__name__}"] = cls


def img_shape_checker(method):
    """Decorator to ensure that the image has always 3 dimensions: WxHC

    Parameters
    ----------
    method : _apply_img method of BaseTransform

    Returns
    -------
    out : method of a class
        Result

    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        res = method(self, *args, **kwargs)
        if len(res.shape) == 2:
            h, w = res.shape
            return res.reshape((h, w, 1))
        elif len(res.shape) != 3:
            raise ValueError

        return res

    return wrapper


def validate_parameter(
    parameter, allowed_modes, default_value, basic_type=str, heritable=True
):
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


def validate_numeric_range_parameter(
    parameter, default_val, min_val=None, max_val=None
):
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

    if not isinstance(parameter, tuple):
        raise TypeError

    if len(parameter) != 2:
        raise ValueError

    if parameter[0] > parameter[1]:
        raise ValueError

    if not (
        isinstance(parameter[0], (int, float))
        and isinstance(parameter[1], (int, float))
    ):
        raise TypeError

    if min_val is not None:
        if parameter[0] < min_val or parameter[1] < min_val:
            raise ValueError

    if max_val is not None:
        if parameter[0] > max_val or parameter[1] > max_val:
            raise ValueError

    return parameter
