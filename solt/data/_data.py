import numpy as np
from ..constants import allowed_types
from ..constants import allowed_interpolations
from ..constants import allowed_paddings
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

            if fmt[idx] == 'I' or fmt[idx] == 'M':
                val = ('nearest', 'strict') if fmt[idx] == 'M' else None
                if 'interpolation' not in transform_settings[idx]:
                    transform_settings[idx]['interpolation'] = validate_parameter(val, allowed_interpolations,
                                                                                  'bilinear', str, True)
                else:
                    transform_settings[idx]['interpolation'] = validate_parameter((
                        transform_settings[idx]['interpolation'], 'strict'),
                        allowed_interpolations,
                        'bilinear', str, True)

                if 'padding' not in transform_settings[idx]:
                    transform_settings[idx]['padding'] = validate_parameter(None, allowed_paddings,
                                                                            'z', str, True)
                else:
                    transform_settings[idx]['padding'] = validate_parameter((
                        transform_settings[idx]['padding'], 'strict'),
                        allowed_paddings,
                        'z', str, True)
            else:
                if 'interpolation' in transform_settings[idx] or 'padding' in transform_settings[idx]:
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
        self.__H = H
        self.__W = W

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
    def H(self):
        return self.__H

    @property
    def W(self):
        return self.__W

    @H.setter
    def H(self, value):
        self.__H = value

    @W.setter
    def W(self, value):
        self.__W = value
