from copy import deepcopy
from ..constants import allowed_types


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
            if not isinstance(data, list):
                data = (data,)

        if not isinstance(data, tuple):
            raise TypeError
        if len(data) != len(fmt):
            raise ValueError

        for t in fmt:
            if t not in allowed_types:
                raise TypeError

        self.__data = deepcopy(data)
        self.__fmt = fmt

    @property
    def data_format(self):
        return self.__fmt

    @property
    def data(self):
        return self.__data

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
