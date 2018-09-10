from functools import wraps
from copy import deepcopy
from .constants import allowed_types


__all__ = ['img_shape_checker', 'DataContainer', 'KeyPoints']


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
        assert 1 < len(res.shape) <= 3
        if len(res.shape) == 2:
            h, w = res.shape
            c = 1
        else:
            h, w, c = res.shape

        return res.reshape((h, w, c))
    return wrapper


class DataContainer(object):
    """Data container to encapsulate different types of data, such as images, bounding boxes, etc.

    The container itself is iterable according to the format.

    Parameters
    ----------
    data : tuple
        Data items stored in a tuple
    fmt : str
        Data format. Example: 'IMMM' - image and three masks.

    """
    def __init__(self, data, fmt):
        if len(fmt) == 1 and not isinstance(data, tuple):
            data = (data,)

        assert isinstance(data, tuple)
        assert len(data) == len(fmt)

        for t in fmt:
            assert t in allowed_types

        self.__data = deepcopy(data)
        self.__fmt = fmt

    @property
    def data_format(self):
        return self.__fmt

    @data_format.setter
    def data_format(self, value):
        self.__fmt = value

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, value):
        self.__fmt = value

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
            Data item (e.g. numpy.ndarray) and its type, e.g. 'I' - image.

        """
        assert isinstance(idx, int)
        return self.__data[idx], self.__fmt[idx]

    def __len__(self):
        return len(self.__data)


class KeyPoints(object):
    """Keypoints class

    Parameters
    ----------
    pts : numpy.ndarray
        Key points as an numpy.ndarray, (x, y) format.
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

    @data.setter
    def data(self, value):
        self.__data = value

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
