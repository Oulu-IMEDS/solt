import json
import yaml
import inspect


class Serializable(object):
    registry = {}
    serializable_name = None
    """How the class should be stored in the registry. If ``None`` (default), the transform is not added."""

    def __init__(self):
        if self.serializable_name is not None:
            self.registry[self.serializable_name] = self.__class__

    def to_dict(self):
        """Method returns a dict, describing the object sufficiently enough to reconstruct it
        back.

        Returns
        -------
        out : dict
            OrderedDict, ready for json serialization.

        """

        d = {}
        argspec = inspect.getfullargspec(self.__class__)

        for item in self.__dict__.items():
            if item[0] not in argspec.args:
                continue
            if hasattr(item[1], "to_dict") and isinstance(item[1], Serializable):
                # Thsi situation is only possible when we serialize the stream
                d[item[0]] = {"stream": item[1].to_dict()}
            elif item[0] != "transforms":
                d[item[0]] = item[1]
            elif isinstance(item[1], (tuple, list)) and item[0] == "transforms":
                d[item[0]] = [{x.__class__.serializable_name: x.to_dict()} for x in item[1]]

        return d

    def to_yaml(self, filename=None):
        """Writes a Serializable object into a file. If the filename is None,
        then this function just returns a string.

        Parameters
        ----------
        filename : str or pathlib.Path or None
            Path to the .yaml file.
        Returns
        -------
        out : str
            Serialized object
        """
        res = yaml.safe_dump({"stream": self.to_dict()})
        if filename is not None:
            with open(filename, "w") as f:
                f.write(res)

        return res

    def to_json(self, filename=None):
        """Writes a Serializable object into a json file. If the filename is None,
        then this function returns a string.


        Parameters
        ----------
        filename : str or pathlib.Path or None

        Returns
        -------
        out : str
            Serialized object

        """
        res = json.dumps({"stream": self.to_dict()}, indent=4)
        if filename is not None:
            with open(filename, "w") as f:
                f.write(res)

        return res

    def __init_subclass__(cls, **kwargs):
        super(Serializable, cls).__init_subclass__(**kwargs)
        if hasattr(cls, "serializable_name"):
            if cls.serializable_name is not None:
                cls.registry[f"{cls.serializable_name}"] = cls


def from_dict(transforms):
    """Deserializes the transformations stored in a dict.
        Supports deserialization of Streams only.

    Parameters
    ----------
    transforms : dict
        Transforms

    Returns
    -------
    out : solt.core.Stream
        An instance of solt.core.Stream.

    """

    if not isinstance(transforms, dict):
        raise TypeError("Transforms must be a dict!")
    for t in transforms:
        if "transforms" in transforms[t]:
            transforms[t]["transforms"] = [from_dict(x) for x in transforms[t]["transforms"]]
        if "affine_transforms" in transforms[t]:
            transforms[t]["affine_transforms"] = from_dict(transforms[t]["affine_transforms"])
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
    s = str(s)

    if s.endswith(".json"):
        with open(s, "r") as f:
            d = json.load(f)
    else:
        d = json.loads(s)

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
    s = str(s)
    if s.endswith(".yaml"):
        with open(s, "r") as f:
            d = yaml.safe_load(f.read())
    else:
        d = yaml.safe_load(s)

    return from_dict(d)
