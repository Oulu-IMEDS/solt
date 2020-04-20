import numpy as np
import torch

from solt.constants import ALLOWED_INTERPOLATIONS, ALLOWED_PADDINGS, ALLOWED_TYPES, IMAGENET_MEAN, IMAGENET_STD
from solt.utils import validate_parameter


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
        Example: ``transform_settings={0:{'interpolation':'bilinear'}, 1: {'interpolation':'bicubic'}}``

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
                        val, ALLOWED_INTERPOLATIONS, "bilinear", str, True
                    )
                else:
                    transform_settings[idx]["interpolation"] = validate_parameter(
                        (transform_settings[idx]["interpolation"], "strict"),
                        ALLOWED_INTERPOLATIONS,
                        "bilinear",
                        str,
                        True,
                    )

                if "padding" not in transform_settings[idx]:
                    transform_settings[idx]["padding"] = validate_parameter(None, ALLOWED_PADDINGS, "z", str, True)
                else:
                    transform_settings[idx]["padding"] = validate_parameter(
                        (transform_settings[idx]["padding"], "strict"), ALLOWED_PADDINGS, "z", str, True,
                    )
            else:
                if "interpolation" in transform_settings[idx] or "padding" in transform_settings[idx]:
                    raise TypeError

        if len(data) != len(transform_settings):
            raise ValueError

        for t in fmt:
            if t not in ALLOWED_TYPES:
                raise TypeError(f"The found type was {t}, but needs to be one of {ALLOWED_TYPES}")

        self.__data = data
        self.__fmt = fmt
        self.__transform_settings = transform_settings

    def validate(self):
        """Validates frame consistency in the wrapped data."""
        frame_prev = tuple()

        for obj, t, settings in self:
            if t == "I":
                frame_curr = obj.shape[:-1]  # exclude channel dim
            elif t == "M":
                frame_curr = obj.shape
            elif t == "P":
                frame_curr = obj.frame
            elif t == "L":
                continue

            if len(frame_prev) == 0:
                frame_prev = frame_curr
            else:
                if len(frame_prev) != len(frame_curr):
                    msg = f"Inconsistent frame dimensionality: " f"{len(frame_prev)}, {len(frame_curr)}"
                    raise ValueError(msg)
                elif frame_prev != frame_curr:
                    msg = f"Inconsistent frames shapes: {frame_prev}, {frame_curr}"
                    raise ValueError(msg)

        return frame_prev

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

        If data is a dict, then the ``solt.data.DataContainer`` will be created so that the
        ``image`` data will be stored first. Subsequently, multiple images under the key ``images``
        will be stored. The same applies to masks (first ``mask`` and then ``masks``),
        labels (``label`` and ``labels``), and the keypoints (``keypoints`` and ``keypoints_array``).
        You must use ``solt.data.KeyPoints`` object here. Labels will always be stored last.

        For example, if the input ``dict`` looks like this: ``d = {'label': l1, 'image': i1, 'mask': m1}`` or
        ``d = {'mask': m1, 'image': i1, 'label': l1}``, the ``DataContainer`` will convert this
        into ``sld.DataContainer((i1, m1, l1), 'IML')``.

        More complex case:

        .. highlight:: python
        .. code-block:: python

            d = {'image': i1, masks: (m1, m2, m3, m4), 'labels': (l1, l2, l3, l4, l5),
            'keypoints': solt.core.KeyPoints(k, h, w)
            dc_from_dict = solt.core.DataContainer.from_dict(d)

        will be equivalent to

        .. highlight:: python
        .. code-block:: python

            dc = solt.core.DataContainer((i1, m1, m2, m3, m4, solt.core.KeyPoints(k, h, w), l1, l2, l3, l4, l5),
            'IMMMMPLLLLLL').


        Please note, that when you create DataContainer using such a simplified interface,
        you cannot setup the transform parameters per item. Use a proper constructor instead.

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

    def wrap_mean_std(self, img, mean, std):
        if not isinstance(mean, (tuple, list, np.ndarray, torch.FloatTensor)):
            msg = (
                f"Unknown type ({type(mean)}) of mean vector! " f"Expected tuple, list, np.ndarray or torch.FloatTensor"
            )
            raise TypeError(msg)

        if not isinstance(std, (tuple, list, np.ndarray, torch.FloatTensor)):
            msg = (
                f"Unknown type ({type(mean)}) of mean vector! " f"Expected tuple, list, np.ndarray or torch.FloatTensor"
            )
            raise TypeError(msg)

        if len(mean) != img.size(0):
            raise ValueError("Mean vector size does not match the number of channels")
        if len(std) != img.size(0):
            raise ValueError("Std vector size does not match the number of channels")

        shape_broadcast = (img.size(0),) + (1,) * (img.ndim - 1)
        if isinstance(mean, (list, tuple)):
            mean = torch.tensor(mean).view(shape_broadcast)
        if isinstance(std, (list, tuple)):
            std = torch.tensor(std).view(shape_broadcast)

        if isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean).view(shape_broadcast).float()
        if isinstance(std, np.ndarray):
            std = torch.from_numpy(std).view(shape_broadcast).float()

        return mean, std

    def to_torch(self, as_dict=False, scale_keypoints=True, normalize=False, mean=None, std=None):
        """This method converts the DataContainer Content into a dict or a list PyTorch objects

        Parameters
        ----------
        as_dict : bool
            Whether to return the result as a dictionary. If a single item is present, then the singular naming
            will be used. If plural, then the plural will be used. The items will be stored and
            sorted a similar manner to the method ``from_dict``: images, masks, keypoints_array, and labels.
            The same applies to a singular case.
        scale_keypoints : bool
            Whether to scale keypoints to 0-1 range. ``True`` by default.
        normalize : bool
            Whether to subtract the `mean` and divide by the `std`.
        mean : torch.Tensor
            Mean to subtract. If None, then the ImageNet mean will be subtracted.
        std : torch.Tensor
            Std to subtract. If None, then the ImageNet std will be subtracted.
        """
        res_dict = {
            "images": list(),
            "masks": list(),
            "keypoints_array": list(),
            "labels": list(),
        }
        not_as_dict = []
        for el, f in zip(self.__data, self.__fmt):
            if f == "I":
                # TODO: remove hardcode. Use np.iinfo, np.finfo
                scale = 255.0
                if el.dtype == np.uint16:
                    scale = 65535.0
                    el = el.astype(np.float32)

                # Move channels to the front to match the PyTorch notation
                img = torch.from_numpy(np.moveaxis(el, -1, 0)).div(scale)

                if normalize:
                    if mean is None or std is None:
                        mean, std = IMAGENET_MEAN, IMAGENET_STD
                    mean, std = self.wrap_mean_std(img, mean, std)
                    img.sub_(mean)
                    img.div_(std)
                res_dict["images"].append(img)
                not_as_dict.append(img)

            elif f == "M":
                mask = torch.from_numpy(el).unsqueeze(0).float()
                res_dict["masks"].append(mask)
                not_as_dict.append(mask)
            elif f == "P":
                landmarks = torch.from_numpy(el.data).float()
                if scale_keypoints:
                    landmarks = landmarks / (torch.tensor(el.frame)[None, ...] - 1)
                res_dict["keypoints_array"].append(landmarks)
                not_as_dict.append(landmarks)
            elif f == "L":
                res_dict["labels"].append(el)
                not_as_dict.append(el)
        if not as_dict:
            if len(not_as_dict) == 1:
                return not_as_dict[0]
            return not_as_dict

        return self.remap_results_dict(res_dict)

    @staticmethod
    def remap_results_dict(res_dict):
        res = {}
        if len(res_dict["images"]) == 1:
            res["image"] = res_dict["images"][0]
        elif len(res_dict["images"]) > 1:
            res["images"] = res_dict["images"]

        if len(res_dict["masks"]) == 1:
            res["mask"] = res_dict["masks"][0]
        elif len(res_dict["masks"]) > 1:
            res["masks"] = res_dict["masks"]

        if len(res_dict["keypoints_array"]) == 1:
            res["keypoints"] = res_dict["keypoints_array"][0]
        elif len(res_dict["keypoints_array"]) > 1:
            res["keypoints_array"] = res_dict["keypoints_array"]

        if len(res_dict["labels"]) == 1:
            res["label"] = res_dict["labels"][0]
        elif len(res_dict["labels"]) > 1:
            res["labels"] = res_dict["labels"]
        return res

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

    def __eq__(self, other):
        fmt_equal = self.data_format == other.data_format
        data_equal = True
        for d1, d2 in zip(self.data, other.data):
            if isinstance(d1, np.ndarray):
                data_equal = data_equal and np.array_equal(d1, d2)
            elif isinstance(d1, Keypoints):
                data_equal = data_equal and (d1 == d2)
            else:
                data_equal = data_equal and d1 == d2

        return fmt_equal and data_equal


class Keypoints(object):
    """Keypoints class

    Parameters
    ----------
    pts : numpy.ndarray
        Array of keypoints. If 2D, has to be in (x, y) format.
    frame : (n, ) list-like of ints
        Shape of the coordinate frame. frame[0] is `height`, frame[1] is `width`.
    """

    def __init__(self, pts=None, frame=None):
        self.__data = pts
        if frame is not None:
            self.__frame = list(frame)
        else:
            if self.__data is None:
                self.__frame = []
            else:
                raise ValueError("Frame is not specified")

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
    def frame(self):
        return tuple(self.__frame)

    @frame.setter
    def frame(self, value):
        self.__frame = value

    def __eq__(self, other):
        dim_equal = all([self.frame[i] == other.frame[i] for i in range(len(self.frame))])
        data_equal = np.array_equal(self.data, other.data)

        return dim_equal and data_equal
