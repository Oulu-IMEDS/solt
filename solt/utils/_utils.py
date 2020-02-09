from functools import wraps
import json
import yaml
import pathlib


class Serializable(object):
    registry = {}
    serializable_name = None

    def __init__(self):
        if self.serializable_name is not None:
            self.registry[self.serializable_name] = self.__class__

    def to_dict(self):
        """Method returns a dict, describing the object sufficiently enough to reconstruct it
        back.

        Returns
        -------
        out : OrderedDict
            OrderedDict, ready for json serialization.

        """

        d = {}
        for item in self.__dict__.items():
            # state_dict is not serialized
            if item[0] == "state_dict":
                continue
            if hasattr(item[1], "to_dict") and isinstance(item[1], Serializable):
                if item[0] != "affine_transforms":
                    raise ValueError(
                        "Found an attribute of the class that is a tranform!"
                    )
                else:
                    d[item[0]] = {"stream": item[1].to_dict()}
            elif item[0] != "transforms":
                d[item[0]] = item[1]
            elif isinstance(item[1], (tuple, list)) and item[0] == "transforms":
                d[item[0]] = [{x.__class__.__name__: x.to_dict()} for x in item[1]]

        return d

    def __init_subclass__(cls, **kwargs):
        super(Serializable, cls).__init_subclass__(**kwargs)
        if hasattr(cls, "serializablie_name"):
            cls.registry[f"{cls.serializablie_name}"] = cls


def from_dict(transforms):
    """Deserializes the transformations stored in a dict.
        Supports deserialization of Streams only.

    Parameters
    ----------
    transforms : dict
        Transforms

    Returns
    -------
    out : slc.Stream
        An instance of solt.Stream

    """

    if not isinstance(transforms, dict):
        raise TypeError("Transforms must be a dict!")
    for t in transforms:
        if "transforms" in transforms[t]:
            transforms[t]["transforms"] = [
                from_dict(x) for x in transforms[t]["transforms"]
            ]
        if "affine_transforms" in transforms[t]:
            transforms[t]["affine_transforms"] = from_dict(
                transforms[t]["affine_transforms"]
            )
        if t in Serializable.registry:
            cls = Serializable.registry[t]
        else:
            raise ValueError(f"Could not find {t} in the registry!")

        return cls(**transforms[t])


def from_json(s):
    """Allows to deserialize transforms from a json file.

    Parameters
    ----------
    s : str or pathlib.Path
        Json string or path. Path can be stored as a string or as a pathlib object.

    Returns
    -------
    s : Serializable
        A serializable object

    """

    d = None
    if isinstance(s, str):
        if s.endswith(".json"):
            with open(s, "r") as f:
                d = json.load(f)
        else:
            d = json.loads(s)
    elif isinstance(s, pathlib.Path):
        if s.suffix == ".json":
            d = json.load(s)
        else:
            raise ValueError("Filename must end with .json")

    return from_dict(d)


def from_yaml(s):
    """Allows to deserialize transforms from a yaml file.

    Parameters
    ----------
    s : str or pathlib.Path
        Path to the yaml object.

    Returns
    -------
    s : Serializable
        A serializable object

    """

    if isinstance(s, str):
        if s.endswith(".yaml"):
            with open(s, "r") as f:
                d = yaml.load(f.read(), Loader=yaml.Loader)
        else:
            d = yaml.load(s, Loader=yaml.Loader)
    elif isinstance(s, pathlib.Path):
        if s.suffix == ".yaml":
            d = yaml.load(s)
        else:
            raise ValueError("File type must end with .yaml")
    else:
        raise TypeError("Input must be a string or a pathlib.Path")

    return from_dict(d)


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
