import numpy as np

from ..constants import allowed_interpolations, allowed_paddings, allowed_types
from ..utils import validate_parameter


class DataContainer(object):
    """Data container to encapsulate different types of data, such as images, bounding boxes, etc.

    The container itself is iterable according to the format.

    Parameters
    ----------
    data : tuple
        Data items stored in a tuple
    fmt : str
        Data format. Example: 'IMMM' - image and three masks.
    transform_settings : dict or None
        Settings for each data item. At this stage, the settings include only padding and interpolation.
        The key in this dict corresponds to the index of the element in the given data tuple.
        The value is another dict, which has all the settings. Segmentation masks have nearest neighbor interpolation
        by default, this can be changed manually if needed.
        Example: transform_settings={0:{'interpolation':'bilinear'}, 1: {'interpolation':'bicubic'}}

    """

    def __init__(self, data, fmt, transform_settings=None):
        if len(fmt) == 1 and not isinstance(data, tuple):
            if not isinstance(data, list):
                data = (data,)
            else:
                raise TypeError

        if not isinstance(data, tuple):
            raise TypeError

        if len(data) != len(fmt):
            raise ValueError

        if transform_settings is not None:
            if not isinstance(transform_settings, dict):
                raise TypeError
        else:
            transform_settings = {}

        # Element-wise settings
        # If no settings provided for certain items, they will be created
        for idx in range(len(data)):
            if idx not in transform_settings:
                transform_settings[idx] = {}

            if fmt[idx] == "I" or fmt[idx] == "M":
                val = ("nearest", "strict") if fmt[idx] == "M" else None
                if "interpolation" not in transform_settings[idx]:
                    transform_settings[idx]["interpolation"] = validate_parameter(
                        val, allowed_interpolations, "bilinear", str, True
                    )
                else:
                    transform_settings[idx]["interpolation"] = validate_parameter(
                        (transform_settings[idx]["interpolation"], "strict"),
                        allowed_interpolations,
                        "bilinear",
                        str,
                        True,
                    )

                if "padding" not in transform_settings[idx]:
                    transform_settings[idx]["padding"] = validate_parameter(
                        None, allowed_paddings, "z", str, True
                    )
                else:
                    transform_settings[idx]["padding"] = validate_parameter(
                        (transform_settings[idx]["padding"], "strict"),
                        allowed_paddings,
                        "z",
                        str,
                        True,
                    )
            else:
                if (
                    "interpolation" in transform_settings[idx]
                    or "padding" in transform_settings[idx]
                ):
                    raise TypeError

        if len(data) != len(transform_settings):
            raise ValueError

        for t in fmt:
            if t not in allowed_types:
                raise TypeError

        self.__data = data
        self.__fmt = fmt
        self.__transform_settings = transform_settings

    @property
    def data_format(self):
        return self.__fmt

    @property
    def data(self):
        return self.__data

    @property
    def transform_settings(self):
        return self.__transform_settings

    @staticmethod
    def from_dict(data):
        """Creates a data container from a dictionary.

        If data is a dict, then the `DataContainer` will be created so that the images stored
        by the key `image` will be stored first. Subsequently, multiple images stored under the key `images`
        will be stored. The same applies to masks (first `mask` and then `masks`), labels,
        and the keypoints (`keypoints` and`keypoints_array`). You must use `solt.data.KeyPoints` object here.
        Labels will always be stored last.

        For example, if the input `dict` looks like this: `d = {'label': l1, 'image': i1, 'mask': m1}` or
        `d = {'mask': m1, 'image': i1, 'label': l1}`, the `DataContainer` will convert this
        into `sld.DataContainer((i1, m1, l1), 'IML')`.

        In a more complex case: `d={'image': i1, masks: (m1, m2, m3, m4), 'labels': (l1, l2, l3, l4, l5),
        'keypoints': sld.KeyPoints(k, h, w)` would be equivalent to
        `sld.DataContainer((i1, m1, m2, m3, m4, sld.KeyPoints(k, h, w), l1, l2, l3, l4, l5), 'IMMMMPLLLLLL')`.

        Please note, that when you create DataContainer using such a simplified interface,
        you cannot setup the transform parameters. Use a proper constructor instead.

        Parameters
        ----------
        data : dict
            Data stored in a dictionary

        Returns
        -------
        out : sld.DataContainer
            Newly instantiated data container object
        """
        dc_content = []
        dc_format = []
        if "image" in data:
            dc_content.append(data["image"])
            dc_format.append("I")
        if "images" in data:
            dc_content.extend(data["images"])
            dc_format.extend("I" * len(data["images"]))
        if "mask" in data:
            dc_content.append(data["mask"])
            dc_format.append("M")
        if "masks" in data:
            dc_content.extend(data["masks"])
            dc_format.extend("M" * len(data["masks"]))
        if "keypoints" in data:
            dc_content.append(data["keypoints"])
            dc_format.append("P")
        if "keypoints_array" in data:
            dc_content.extend(data["keypoints_array"])
            dc_format.extend("P" * len(data["keypoints_array"]))
        if "label" in data:
            dc_content.append(data["label"])
            dc_format.append("L")
        if "labels" in data:
            dc_content.extend(data["labels"])
            dc_format.extend("L" * len(data["labels"]))

        return DataContainer(tuple(dc_content), "".join(dc_format))

    def __getitem__(self, idx):
        """
        Returns a data item and its type using index.

        Parameters
        ----------
        idx : int
            Index of an element to return according to the specified format

        Returns
        -------
        out : tuple
            Data item (e.g. numpy.ndarray), its type, e.g. 'I' - image and the transform settings,
            e.g. interpolation

        """
        return self.__data[idx], self.__fmt[idx], self.__transform_settings[idx]

    def __len__(self):
        return len(self.__data)


class KeyPoints(object):
    """Keypoints class

    Parameters
    ----------
    pts : numpy.ndarray
        Key points as an numpy.ndarray in (x, y) format.
    H : int
        Height of the coordinate frame.
    W : int
        Width of the coordinate frame.
    """

    def __init__(self, pts=None, H=None, W=None):
        self.__data = pts
        self.__height = H
        self.__width = W

    @property
    def data(self):
        return self.__data

    def __getitem__(self, idx):
        return self.__data[idx, :]

    def __setitem__(self, idx, value):
        if not isinstance(value, np.ndarray):
            raise TypeError
        self.__data[idx, :] = value

    @property
    def height(self):
        return self.__height

    @property
    def width(self):
        return self.__width

    @height.setter
    def height(self, value):
        self.__height = value

    @width.setter
    def width(self, value):
        self.__width = value
