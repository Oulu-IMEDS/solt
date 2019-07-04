import random

import cv2
import numpy as np

from ..base_transforms import BaseTransform, MatrixTransform, \
    PaddingPropertyHolder, DataDependentSamplingTransform, InterpolationPropertyHolder
from ..base_transforms import ImageTransform
from ..constants import allowed_paddings, allowed_crops, \
    dtypes_max, allowed_blurs, allowed_color_conversions, allowed_interpolations
from ..core import Stream
from ..data import KeyPoints, DataContainer
from ..utils import img_shape_checker
from ..utils import validate_parameter, validate_numeric_range_parameter


class RandomFlip(BaseTransform):
    """Random Flipping transform.

    Parameters
    ----------
    p : float
        Probability of flip
    axis : int
        Axis of flip. Here, 1 stands for horizontal flipping, 0 stands for the vertical one. -1 stands for
        both axes.
    """

    def __init__(self, p=0.5, axis=1, data_indices=None):

        super(RandomFlip, self).__init__(p=p, data_indices=data_indices)
        if axis not in [-1, 0, 1]:
            raise ValueError

        self.__axis = axis

    def sample_transform(self):
        pass

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        if self.__axis == 0:
            return np.ascontiguousarray(img[::-1, ...])
        elif self.__axis == 1:
            return np.ascontiguousarray(img[:, ::-1, ...])
        else:
            return np.ascontiguousarray(img[::-1, ::-1, ...])

    def _apply_mask(self, mask: np.ndarray, settings: dict):
        mask_new = cv2.flip(mask, self.__axis)
        return mask_new

    def _apply_labels(self, labels, settings: dict):
        return labels

    def _apply_pts(self, pts: KeyPoints, settings: dict):
        # We should guarantee that we do not change the original data.
        pts_data = pts.data.copy()
        if self.__axis == 0:
            pts_data[:, 1] = pts.H - 1 - pts_data[:, 1]
        elif self.__axis == 1:
            pts_data[:, 0] = pts.W - 1 - pts_data[:, 0]
        elif self.__axis == -1:
            pts_data[:, 1] = pts.H - 1 - pts_data[:, 1]
            pts_data[:, 0] = pts.W - 1 - pts_data[:, 0]

        return KeyPoints(pts=pts_data, H=pts.H, W=pts.W)


class RandomRotate(MatrixTransform):
    """Random rotation around the center clockwise

    Parameters
    ----------
    rotation_range : tuple or float or None
        Range of rotation.
        If float, then (-rotation_range, rotation_range) will be used for transformation sampling.
        if None, then rotation_range=(0,0).
    interpolation : str or tuple or None
        Interpolation type. Check the allowed interpolation types.
    padding : str or tuple or None
        Padding mode. Check the allowed padding modes.
    p : float
        Probability of using this transform
    ignore_state : bool
        Whether to ignore the state. See details in the docs for `MatrixTransform`.

    """

    def __init__(self, rotation_range=None, interpolation='bilinear', padding='z', p=0.5, ignore_state=True):
        super(RandomRotate, self).__init__(interpolation=interpolation, padding=padding,
                                           p=p, ignore_state=ignore_state)
        if isinstance(rotation_range, (int, float)):
            rotation_range = (-rotation_range, rotation_range)
        self.__range = validate_numeric_range_parameter(rotation_range, (0, 0))

    @property
    def rotation_range(self):
        return self.__range

    def sample_transform(self):
        """
        Samples random rotation within specified range and saves it as an object state.

        """
        rot = np.deg2rad(random.uniform(self.__range[0], self.__range[1]))

        self.state_dict['rot'] = rot

        self.state_dict['transform_matrix'][0, 0] = np.cos(rot)
        self.state_dict['transform_matrix'][0, 1] = -np.sin(rot)
        self.state_dict['transform_matrix'][0, 2] = 0

        self.state_dict['transform_matrix'][1, 0] = np.sin(rot)
        self.state_dict['transform_matrix'][1, 1] = np.cos(rot)
        self.state_dict['transform_matrix'][1, 2] = 0

        self.state_dict['transform_matrix'][2, 0] = 0
        self.state_dict['transform_matrix'][2, 1] = 0
        self.state_dict['transform_matrix'][2, 2] = 1


class RandomRotate90(RandomRotate):
    """Random rotation around the center by 90 degrees.

    Parameters
    ----------
    k : int
        How many times to rotate the data. If positive, indicates the clockwise direction.
        Zero by default.
    p : float
        Probability of using this transform

    """

    def __init__(self, k=0, p=0.5):
        if not isinstance(k, int):
            raise TypeError
        super(RandomRotate90, self).__init__(p=p, rotation_range=(k * 90, k * 90))
        self.__k = -k

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        return np.ascontiguousarray(np.rot90(img, self.__k))

    def _apply_mask(self, mask: np.ndarray, settings: dict):
        return np.ascontiguousarray(np.rot90(mask, self.__k))


class RandomShear(MatrixTransform):
    """Random shear around the center.

    Parameters
    ----------
    range_x : tuple or float
        Shearing range along X-axis.
        If float, then (-range_x, range_x) will be used.
        If None, then range_x=(0, 0)
    range_y : tuple or float or None
        Shearing range along Y-axis. If float, then (-range_y, range_y) will be used.
        If None, then range_y=(0, 0)
    interpolation : str or tuple or None or tuple or None
        Interpolation type. Check the allowed interpolation types.
    padding : str
        Padding mode. Check the allowed padding modes.
    p : float
        Probability of using this transform
    ignore_state : bool
        Whether to ignore the state. See details in the docs for `MatrixTransform`.

    """

    def __init__(self, range_x=None, range_y=None, interpolation='bilinear',
                 padding='z', p=0.5, ignore_state=True):
        super(RandomShear, self).__init__(p=p, padding=padding, interpolation=interpolation, ignore_state=ignore_state)
        if isinstance(range_x, (int, float)):
            range_x = (-range_x, range_x)

        if isinstance(range_y, (int, float)):
            range_y = (-range_y, range_y)

        self.__range_x = validate_numeric_range_parameter(range_x, (0, 0))
        self.__range_y = validate_numeric_range_parameter(range_y, (0, 0))

    @property
    def shear_range_x(self):
        return self.__range_x

    @property
    def shear_range_y(self):
        return self.__range_y

    def sample_transform(self):
        shear_x = random.uniform(self.shear_range_x[0], self.shear_range_x[1])
        shear_y = random.uniform(self.shear_range_y[0], self.shear_range_y[1])

        self.state_dict['shear_x'] = shear_x
        self.state_dict['shear_y'] = shear_y

        self.state_dict['transform_matrix'][0, 0] = 1
        self.state_dict['transform_matrix'][0, 1] = shear_x
        self.state_dict['transform_matrix'][0, 2] = 0

        self.state_dict['transform_matrix'][1, 0] = shear_y
        self.state_dict['transform_matrix'][1, 1] = 1
        self.state_dict['transform_matrix'][1, 2] = 0

        self.state_dict['transform_matrix'][2, 0] = 0
        self.state_dict['transform_matrix'][2, 1] = 0
        self.state_dict['transform_matrix'][2, 2] = 1


class RandomScale(MatrixTransform):
    """Random scale transform.

    Parameters
    ----------
    range_x : tuple or float or None
        Scaling range along X-axis.
        If float, then (min(1, range_x), max(1, range_x)) will be used.
        If None, then range_x =(1,1) by default.
    range_y : tuple or float
        Scaling range along Y-axis.
        If float, then (min(1, range_y), max(1, range_y)) will be used.
        If None, then range_y=(1,1) by default.
    same: bool
        Indicates whether to use the same scaling factor for width and height.
    interpolation : str or tuple or None
        Interpolation type. Check the allowed interpolation types.
        one indicates default behavior - bilinear mode.
    p : float
        Probability of using this transform
    ignore_state : bool
        Whether to ignore the state. See details in the docs for `MatrixTransform`.

    """

    def __init__(self, range_x=None, range_y=None, same=True, interpolation='bilinear', p=0.5, ignore_state=True):

        super(RandomScale, self).__init__(interpolation=interpolation, padding=None, p=p, ignore_state=ignore_state)

        if isinstance(range_x, (int, float)):
            range_x = (min(1, range_x), max(1, range_x))

        if isinstance(range_y, (int, float)):
            range_y = (min(1, range_y), max(1, range_y))

        self.__same = same
        self.__range_x = None if range_x is None else validate_numeric_range_parameter(range_x, (1, 1), min_val=0)
        self.__range_y = None if range_y is None else validate_numeric_range_parameter(range_y, (1, 1), min_val=0)

    @property
    def scale_range_x(self):
        return self.__range_x

    @property
    def scale_range_y(self):
        return self.__range_y

    def sample_transform(self):
        if self.scale_range_x is None:
            scale_x = 1
        else:
            scale_x = random.uniform(self.scale_range_x[0], self.scale_range_x[1])

        if self.scale_range_y is None:
            scale_y = 1
        else:
            scale_y = random.uniform(self.scale_range_y[0], self.scale_range_y[1])

        if self.__same:
            if self.__range_x is None:
                scale_x = scale_y
            else:
                scale_y = scale_x

        self.state_dict['scale_x'] = scale_x
        self.state_dict['scale_y'] = scale_y

        self.state_dict['transform_matrix'][0, 0] = scale_x
        self.state_dict['transform_matrix'][0, 1] = 0
        self.state_dict['transform_matrix'][0, 2] = 0

        self.state_dict['transform_matrix'][1, 0] = 0
        self.state_dict['transform_matrix'][1, 1] = scale_y
        self.state_dict['transform_matrix'][1, 2] = 0

        self.state_dict['transform_matrix'][2, 0] = 0
        self.state_dict['transform_matrix'][2, 1] = 0
        self.state_dict['transform_matrix'][2, 2] = 1


class RandomTranslate(MatrixTransform):
    """Random Translate transform..

    Parameters
    ----------
    range_x: tuple or int or None
        Translation range along the horizontal axis. If int, then range_x=(-range_x, range_x).
        If None, then range_x=(0,0).
    range_y: tuple or int or None
        Translation range along the vertical axis. If int, then range_y=(-range_y, range_y).
        If None, then range_y=(0,0).
    interpolation: str
        Interpolation type. See allowed_interpolations in constants.
    padding: str
        Padding mode. See allowed_paddings  in constants
    p: float
        probability of applying this transform.
    """

    def __init__(self, range_x=None, range_y=None, interpolation='bilinear', padding='z', p=0.5, ignore_state=True):
        super(RandomTranslate, self).__init__(interpolation=interpolation, padding=padding,
                                              p=p, ignore_state=ignore_state)
        if isinstance(range_x, (int, float)):
            range_x = (min(range_x, -range_x), max(range_x, -range_x))

        if isinstance(range_y, (int, float)):
            range_y = (min(range_y, -range_y), max(range_y, -range_y))

        self.__range_x = validate_numeric_range_parameter(range_x, (0, 0))
        self.__range_y = validate_numeric_range_parameter(range_y, (0, 0))

    @property
    def translate_range_x(self):
        return self.__range_x

    @property
    def translate_range_y(self):
        return self.__range_y

    def sample_transform(self):
        tx = random.uniform(self.translate_range_x[0], self.translate_range_x[1])
        ty = random.uniform(self.translate_range_y[0], self.translate_range_y[1])

        self.state_dict['translate_x'] = tx
        self.state_dict['translate_y'] = ty

        self.state_dict['transform_matrix'][0, 0] = 1
        self.state_dict['transform_matrix'][0, 1] = 0
        self.state_dict['transform_matrix'][0, 2] = tx

        self.state_dict['transform_matrix'][1, 0] = 0
        self.state_dict['transform_matrix'][1, 1] = 1
        self.state_dict['transform_matrix'][1, 2] = ty

        self.state_dict['transform_matrix'][2, 0] = 0
        self.state_dict['transform_matrix'][2, 1] = 0
        self.state_dict['transform_matrix'][2, 2] = 1


class RandomProjection(MatrixTransform):
    """Random Projective transform.

    Takes a set of affine transforms.

    Parameters
    ----------
    affine_transforms : Stream or None
        Stream object, which has a parameterized Affine Transform.
        If None, then a zero degrees rotation matrix is instantiated.
    v_range : tuple or None
        Projective parameters range. If None, then v_range = (0, 0)
    p : float
        Probability of using this transform.
    """

    def __init__(self, affine_transforms=None, v_range=None, interpolation='bilinear', padding='z',
                 p=0.5, ignore_state=True):

        super(RandomProjection, self).__init__(interpolation=interpolation, padding=padding,
                                               p=p, ignore_state=ignore_state)

        if affine_transforms is None:
            affine_transforms = Stream([
                RandomRotate(rotation_range=0, interpolation=interpolation, p=1)
            ])

        if not isinstance(affine_transforms, Stream):
            raise TypeError
        for trf in affine_transforms.transforms:
            if not isinstance(trf, MatrixTransform):
                raise TypeError

        self.__affine_transforms = affine_transforms
        self.__vrange = validate_numeric_range_parameter(v_range, (0, 0))  # projection components.

    def sample_transform(self):
        if len(self.__affine_transforms.transforms) > 1:
            trf = Stream.optimize_stack(self.__affine_transforms.transforms)
            if len(trf) == 0:
                trf = None
            else:
                trf = trf[0]
        elif len(self.__affine_transforms.transforms) == 0:
            trf = None
        else:
            trf = self.__affine_transforms.transforms[0]
            trf.sample_transform()

        if trf is None:
            M = np.eye(3)
        else:
            M = trf.state_dict['transform_matrix'].copy()

        M[-1, 0] = random.uniform(self.__vrange[0], self.__vrange[1])
        M[-1, 1] = random.uniform(self.__vrange[0], self.__vrange[1])
        self.state_dict['transform_matrix'] = M


class PadTransform(DataDependentSamplingTransform, PaddingPropertyHolder):
    """Transformation, which pads the input to a given size

    Parameters
    ----------
    pad_to : tuple or int
        Target size (W_new, Y_new). The padding is computed using teh following equations:

        left_pad = (pad_to[0] - w) // 2
        right_pad = (pad_to[0] - w) // 2 + (pad_to[0] - w) % 2
        top_pad = (pad_to[1] - h) // 2
        bottom_pad = (pad_to[1] - h) // 2 + (pad_to[1] - h) % 2
    padding :
        Padding type.

    """

    def __init__(self, pad_to, padding=None):
        DataDependentSamplingTransform.__init__(self, p=1)
        PaddingPropertyHolder.__init__(self, padding)
        if not isinstance(pad_to, tuple) and not isinstance(pad_to, int):
            raise TypeError
        if isinstance(pad_to, int):
            pad_to = (pad_to, pad_to)

        self._pad_to = pad_to

    def sample_transform(self):
        DataDependentSamplingTransform.sample_transform(self)

    def sample_transform_from_data(self, data: DataContainer):
        h, w = DataDependentSamplingTransform.sample_transform_from_data(self, data)

        pad_w = (self._pad_to[0] - w) // 2
        pad_h = (self._pad_to[1] - h) // 2

        pad_h_top = pad_h
        pad_h_bottom = pad_h + (self._pad_to[1] - h) % 2

        pad_w_left = pad_w
        pad_w_right = pad_w + (self._pad_to[0] - w) % 2

        if pad_h < 0:
            pad_h_top = 0
            pad_h_bottom = 0

        if pad_w < 0:
            pad_w_left = 0
            pad_w_right = 0

        self.state_dict = {'pad_h': (pad_h_top, pad_h_bottom), 'pad_w': (pad_w_left, pad_w_right)}

    def _apply_img_or_mask(self, img: np.ndarray, settings: dict):
        pad_h_top, pad_h_bottom = self.state_dict['pad_h']
        pad_w_left, pad_w_right = self.state_dict['pad_w']
        padding = allowed_paddings[self.padding[0]]

        if settings['padding'][1] == 'strict':
            padding = allowed_paddings[settings['padding'][0]]

        return cv2.copyMakeBorder(img, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right, padding, value=0)

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        return self._apply_img_or_mask(img, settings)

    def _apply_mask(self, mask: np.ndarray, settings: dict):
        return self._apply_img_or_mask(mask, settings)

    def _apply_labels(self, labels, settings: dict):
        return labels

    def _apply_pts(self, pts: KeyPoints, settings: dict):
        if self.padding[0] != 'z':
            raise ValueError
        pts_data = pts.data.copy()

        pad_h_top, pad_h_bottom = self.state_dict['pad_h']
        pad_w_left, pad_w_right = self.state_dict['pad_w']

        pts_data[:, 0] += pad_w_left
        pts_data[:, 1] += pad_h_top

        return KeyPoints(pts_data, pad_h_top + pts.H + pad_h_bottom,
                         pad_w_left + pts.W + pad_w_right)


class ResizeTransform(BaseTransform, InterpolationPropertyHolder):
    """Transformation, which resizes the input to a given size

    Parameters
    ----------
    resize_to : tuple or int
        Target size (W_new, Y_new).
    interpolation :
        Interpolation type.

    """

    def __init__(self, resize_to, interpolation='bilinear'):
        BaseTransform.__init__(self, p=1)
        InterpolationPropertyHolder.__init__(self, interpolation=interpolation)
        if not isinstance(resize_to, tuple) and not isinstance(resize_to, int):
            raise TypeError
        if isinstance(resize_to, int):
            resize_to = (resize_to, resize_to)

        self._resize_to = resize_to

    def sample_transform(self):
        pass

    def _apply_img_or_mask(self, img: np.ndarray, settings: dict):
        interp = allowed_interpolations[self.interpolation[0]]
        if settings['interpolation'][1] == 'strict':
            interp = allowed_interpolations[settings['interpolation'][0]]

        return cv2.resize(img, self._resize_to, interpolation=interp)

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        return self._apply_img_or_mask(img, settings)

    def _apply_mask(self, mask: np.ndarray, settings: dict):
        return self._apply_img_or_mask(mask, settings)

    def _apply_labels(self, labels, settings: dict):
        return labels

    def _apply_pts(self, pts: KeyPoints, settings: dict):
        pts_data = pts.data.copy().astype(float)

        resize_x, resize_y = self._resize_to

        scale_x = resize_x / pts.W
        scale_y = resize_y / pts.H

        pts_data[:, 0] *= scale_x
        pts_data[:, 1] *= scale_y

        pts_data = pts_data.astype(int)

        return KeyPoints(pts_data, resize_y, resize_x)


class CropTransform(DataDependentSamplingTransform):
    """Center / Random crop transform.

    Object performs center or random cropping depending on the parameters.

    Parameters
    ----------
    crop_size : tuple or int
        Size of the crop (W_new, H_new). If int, then a square crop will be made.
    crop_mode : str
        Crop mode. Can be either 'c' - center or 'r' - random.

    """

    def __init__(self, crop_size, crop_mode='c'):
        super(CropTransform, self).__init__(p=1, data_indices=None)

        if not isinstance(crop_size, int) and not isinstance(crop_size, tuple):
            raise TypeError
        if crop_mode not in allowed_crops:
            raise ValueError

        if isinstance(crop_size, tuple):
            if not isinstance(crop_size[0], int) or not isinstance(crop_size[1], int):
                raise TypeError

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        self._crop_size = crop_size
        self._crop_mode = crop_mode

    @property
    def crop_mode(self):
        return self._crop_mode

    @property
    def crop_size(self):
        return self._crop_size

    def sample_transform(self):
        raise NotImplementedError

    def sample_transform_from_data(self, data: DataContainer):
        h, w = DataDependentSamplingTransform.sample_transform_from_data(self, data)

        if self.crop_size[0] > w or self.crop_size[1] > h:
            raise ValueError

        if self.crop_mode == 'r':
            self.state_dict['x'] = int(random.random() * (w - self.crop_size[0]))
            self.state_dict['y'] = int(random.random() * (h - self.crop_size[1]))

        else:
            self.state_dict['x'] = w // 2 - self.crop_size[0] // 2
            self.state_dict['y'] = h // 2 - self.crop_size[1] // 2

    def __crop_img_or_mask(self, img_mask):
        return img_mask[self.state_dict['y']:self.state_dict['y'] + self.crop_size[1],
                        self.state_dict['x']:self.state_dict['x'] + self.crop_size[0]]

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        return self.__crop_img_or_mask(img)

    def _apply_mask(self, mask: np.ndarray, settings: dict):
        return self.__crop_img_or_mask(mask)

    def _apply_labels(self, labels, settings: dict):
        return labels

    def _apply_pts(self, pts: KeyPoints, settings: dict):
        pts_data = pts.data.copy()
        x, y = self.state_dict['x'], self.state_dict['y']

        pts_data[:, 0] -= x
        pts_data[:, 1] -= y

        return KeyPoints(pts_data, self.crop_size[1], self.crop_size[0])


class ImageAdditiveGaussianNoise(DataDependentSamplingTransform):
    """Adds noise to an image. Other types of data than the image are ignored.

    Parameters
    ----------
    p : float
        Probability of applying this transform,
    gain_range : tuple or float or None
        Gain of the noise. Final image is created as (1-gain)*img + gain*noise.
        If float, then gain_range = (0, gain_range). If None, then gain_range=(0, 0).
    data_indices : tuple or None
        Indices of the images within the data container to which this transform needs to be applied.
        Every element within the tuple must be integer numbers.
        If None, then the transform will be applied to all the images withing the DataContainer.

    """

    def __init__(self, p=0.5, gain_range=0.1, data_indices=None):
        super(ImageAdditiveGaussianNoise, self).__init__(p=p, data_indices=data_indices)
        if isinstance(gain_range, float):
            gain_range = (0, gain_range)

        self._gain_range = validate_numeric_range_parameter(gain_range, (0, 0), min_val=0, max_val=1)

    def sample_transform(self):
        raise NotImplementedError

    def sample_transform_from_data(self, data: DataContainer):
        DataDependentSamplingTransform.sample_transform_from_data(self, data)
        gain = random.uniform(self._gain_range[0], self._gain_range[1])
        h = None
        w = None
        c = None
        obj = None
        for obj, t, _ in data:
            if t == 'I':
                h = obj.shape[0]
                w = obj.shape[1]
                c = obj.shape[2]
                break

        if w is None or h is None or c is None:
            raise ValueError

        random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
        noise_img = random_state.randn(h, w, c)

        noise_img -= noise_img.min()
        noise_img /= noise_img.max()
        noise_img *= 255
        noise_img = noise_img.astype(obj.dtype)

        self.state_dict = {'noise': noise_img, 'gain': gain}

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        return cv2.addWeighted(img, (1 - self.state_dict['gain']),
                               self.state_dict['noise'], self.state_dict['gain'], 0)

    def _apply_mask(self, mask: np.ndarray, settings: dict):
        return mask

    def _apply_labels(self, labels, settings: dict):
        return labels

    def _apply_pts(self, pts: KeyPoints, settings: dict):
        return pts


class ImageCutOut(ImageTransform, DataDependentSamplingTransform):
    """Does cutout augmentation.

    https://arxiv.org/abs/1708.04552

    Parameters
    ----------
    p : float
        Probability of applying this transform,
    cutout_size : tuple or int or None
        The size of the cutout. If None, then it is equal to 2.
    data_indices : tuple or None
        Indices of the images within the data container to which this transform needs to be applied.
        Every element within the tuple must be integer numbers.
        If None, then the transform will be applied to all the images withing the DataContainer.

    """

    def __init__(self, p=0.5, cutout_size=2, data_indices=None):
        super(ImageCutOut, self).__init__(p=p, data_indices=data_indices)
        if not isinstance(cutout_size, int) and not isinstance(cutout_size, tuple):
            raise TypeError

        if isinstance(cutout_size, tuple):
            if not isinstance(cutout_size[0], int) or not isinstance(cutout_size[1], int):
                raise TypeError

        if isinstance(cutout_size, int):
            cutout_size = (cutout_size, cutout_size)

        self._cutout_size = cutout_size

    def sample_transform(self):
        raise NotImplementedError

    def sample_transform_from_data(self, data: DataContainer):
        h, w = DataDependentSamplingTransform.sample_transform_from_data(self, data)

        if self._cutout_size[0] > w or self._cutout_size[1] > h:
            raise ValueError

        self.state_dict['x'] = int(random.random() * (w - self._cutout_size[0]))
        self.state_dict['y'] = int(random.random() * (h - self._cutout_size[1]))

    def __cutout_img(self, img):
        img[self.state_dict['y']:self.state_dict['y'] + self._cutout_size[1],
            self.state_dict['x']:self.state_dict['x'] + self._cutout_size[0]] = 0
        return img

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        return self.__cutout_img(img)


class ImageSaltAndPepper(ImageTransform, DataDependentSamplingTransform):
    """Adds salt and pepper noise to an image. Other types of data than the image are ignored.

    Parameters
    ----------
    p : float
        Probability of applying this transform,
    gain_range : tuple or float or None
        Gain of the noise. Indicates percentage of indices, which will be changed.
        If float, then gain_range = (0, gain_range).
    salt_p : float or tuple or None
        Percentage of salt. Percentage of pepper is 1-salt_p. If tuple, then salt_p is chosen uniformly from the
        given range.
    data_indices : tuple or None
        Indices of the images within the data container to which this transform needs to be applied.
        Every element within the tuple must be integer numebers.
        If None, then the transform will be applied to all the images withing the DataContainer.

    """

    def __init__(self, p=0.5, gain_range=0.1, salt_p=0.5, data_indices=None):
        super(ImageSaltAndPepper, self).__init__(p=p, data_indices=data_indices)
        if not isinstance(gain_range, float) and not isinstance(gain_range, tuple):
            raise TypeError
        if not isinstance(salt_p, float) and not isinstance(salt_p, tuple):
            raise TypeError

        if isinstance(gain_range, float):
            gain_range = (0, gain_range)

        if isinstance(salt_p, float):
            salt_p = (salt_p, salt_p)

        self._gain_range = validate_numeric_range_parameter(gain_range, (0, 0), 0, 1)
        self._salt_p = validate_numeric_range_parameter(salt_p, (0, 0), 0, 1)

    def sample_transform(self):
        raise NotImplementedError

    def sample_transform_from_data(self, data: DataContainer):
        h, w = DataDependentSamplingTransform.sample_transform_from_data(self, data)
        gain = random.uniform(self._gain_range[0], self._gain_range[1])
        salt_p = random.uniform(self._salt_p[0], self._salt_p[1])

        random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
        sp = random_state.rand(h, w) <= gain
        salt = sp.copy() * 1.
        pepper = sp.copy() * 1.
        salt_mask = (random_state.rand(sp.sum()) <= salt_p)
        pepper_mask = 1 - salt_mask
        salt[np.where(salt)] *= salt_mask
        pepper[np.where(pepper)] *= pepper_mask

        self.state_dict = {'salt': salt, 'pepper': pepper}

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        img = img.copy()
        img[np.where(self.state_dict['salt'])] = dtypes_max[img.dtype]
        img[np.where(self.state_dict['pepper'])] = 0
        return img


class ImageGammaCorrection(ImageTransform):
    """Transform applies random gamma correction

    Parameters
    ----------
    p : float
        Probability of applying this transform,
    gamma_range : tuple or float or None
        Gain of the noise. Indicates percentage of indices, which will be changed.
        If float, then gain_range = (1-gamma_range, 1+gamma_range).
    data_indices : tuple or None
        Indices of the images within the data container to which this transform needs to be applied.
        Every element within the tuple must be integer numbers.
        If None, then the transform will be applied to all the images withing the DataContainer.

    """

    def __init__(self, p=0.5, gamma_range=0.1, data_indices=None):
        super(ImageGammaCorrection, self).__init__(p=p, data_indices=data_indices)

        if not isinstance(gamma_range, float) and not isinstance(gamma_range, tuple):
            raise TypeError

        if isinstance(gamma_range, float):
            gamma_range = (1 - gamma_range, 1 + gamma_range)

        self._gamma_range = validate_numeric_range_parameter(gamma_range, (1, 1), 0)

    def sample_transform(self):
        gamma = random.uniform(self._gamma_range[0], self._gamma_range[1])
        inv_gamma = 1.0 / gamma
        lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

        self.state_dict = {'gamma': gamma, 'LUT': lut}

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        return cv2.LUT(img, self.state_dict['LUT'])


class ImageRandomContrast(ImageTransform):
    """Transform randomly changes the contrast

    Parameters
    ----------
    p : float
        Probability of applying this transform,
    contrast_range : tuple or float or None
        Gain of the noise. Indicates percentage of indices, which will be changed.
        If float, then gain_range = (1-contrast_range, 1+contrast_range).
    data_indices : tuple or None
        Indices of the images within the data container to which this transform needs to be applied.
        Every element within the tuple must be integer numbers.
        If None, then the transform will be applied to all the images withing the DataContainer.

    """

    def __init__(self, p=0.5, contrast_range=0.1, data_indices=None):
        super(ImageRandomContrast, self).__init__(p=p, data_indices=data_indices)

        if not isinstance(contrast_range, float) and not isinstance(contrast_range, tuple):
            raise TypeError

        if isinstance(contrast_range, float):
            contrast_range = (1 - contrast_range, 1 + contrast_range)

        self._contrast_range = validate_numeric_range_parameter(contrast_range, (1, 1), 0)

    def sample_transform(self):
        contrast_mul = random.uniform(self._contrast_range[0], self._contrast_range[1])
        lut = np.arange(0, 256) * contrast_mul
        lut = np.clip(lut, 0, 255).astype("uint8")
        self.state_dict = {'contrast_mul': contrast_mul, 'LUT': lut}

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        return cv2.LUT(img, self.state_dict['LUT'])


class ImageBlur(ImageTransform):
    """Transform blurs an image

    Parameters
    ----------
    p : float
        Probability of applying this transform,
    blur_type : str
        Blur type. Allowed blurs
    k_size: int or tuple
        Kernel sizes of the blur. if int, then sampled from (k_size, k_size). If tuple,
        then sampled from the whole tuple. All the values here must be odd.
    gaussian_sigma: int or float or tuple
        Gaussian sigma value. Used for both X and Y axes. If None, then gaussian_sigma=1.
    data_indices : tuple or None
        Indices of the images within the data container to which this transform needs to be applied.
        Every element within the tuple must be integer numbers.
        If None, then the transform will be applied to all the images withing the DataContainer.

    """

    def __init__(self, p=0.5, blur_type=None, k_size=3, gaussian_sigma=None, data_indices=None):
        super(ImageBlur, self).__init__(p=p, data_indices=data_indices)
        if not isinstance(k_size, (int, tuple)):
            raise TypeError

        if isinstance(k_size, int):
            k_size = (k_size, k_size)

        for k in k_size:
            if k % 2 == 0 or k < 1 or not isinstance(k, int):
                raise ValueError

        if isinstance(gaussian_sigma, (int, float)):
            gaussian_sigma = (gaussian_sigma, gaussian_sigma)

        self._blur = validate_parameter(blur_type, allowed_blurs, 'g', basic_type=str, heritable=False)
        self._k_size = k_size
        self._gaussian_sigma = validate_numeric_range_parameter(gaussian_sigma, (1, 1), 0)

    def sample_transform(self):
        k = random.choice(self._k_size)
        s = random.uniform(self._gaussian_sigma[0], self._gaussian_sigma[1])
        self.state_dict = {'k_size': k, 'sigma': s}

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        if self._blur == 'g':
            return cv2.GaussianBlur(img,
                                    ksize=(self.state_dict['k_size'], self.state_dict['k_size']),
                                    sigmaX=self.state_dict['sigma'])
        if self._blur == 'm':
            return cv2.medianBlur(img, ksize=self.state_dict['k_size'])


class ImageRandomHSV(ImageTransform):
    """Performs a random HSV color shift.

    Parameters
    ----------
    p : float
        Probability of applying this transform,
    h_range: tuple or None
        Hue shift range. If None, than h_range=(0, 0).
    s_range: tuple or None
        Saturation shift range. If None, then s_range=(0, 0).
    v_range: tuple or None
        Value shift range. If None, then v_range=(0, 0).
    data_indices : tuple or None
        Indices of the images within the data container to which this transform needs to be applied.
        Every element within the tuple must be integer numbers.
        If None, then the transform will be applied to all the images withing the DataContainer.

    """

    def __init__(self, p=0.5, h_range=None, s_range=None, v_range=None, data_indices=None):
        super(ImageRandomHSV, self).__init__(p=p, data_indices=data_indices)
        self._h_range = validate_numeric_range_parameter(h_range, (0, 0))
        self._s_range = validate_numeric_range_parameter(s_range, (0, 0))
        self._v_range = validate_numeric_range_parameter(v_range, (0, 0))

    def sample_transform(self):
        h = random.uniform(self._h_range[0], self._h_range[1])
        s = random.uniform(self._s_range[0], self._s_range[1])
        v = random.uniform(self._v_range[0], self._v_range[1])
        self.state_dict = {'h_mod': h, 's_mod': s, 'v_mod': v}

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        img = img.copy()
        dtype = img.dtype
        if img.shape[-1] != 3:
            raise ValueError

        if dtype != np.uint8:
            raise TypeError

        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(img_hsv.astype(np.int32))

        h = np.clip(abs((h + self.state_dict['h_mod']) % 180), 0, 180).astype(dtype)
        s = np.clip(s + self.state_dict['s_mod'], 0, 255).astype(dtype)
        v = np.clip(v + self.state_dict['v_mod'], 0, 255).astype(dtype)

        img_hsv_shifted = cv2.merge((h, s, v))
        img = cv2.cvtColor(img_hsv_shifted, cv2.COLOR_HSV2RGB)

        return img


class ImageRandomBrightness(ImageTransform):
    """Performs a random brightness augmentation

    Parameters
    ----------
    p : float
        Probability of applying this transform,
    brightness_range: tuple or None
        brightness_range shift range. If None, then brightness_range=(0, 0).
    data_indices : tuple or None
        Indices of the images within the data container to which this transform needs to be applied.
        Every element within the tuple must be integer numbers.
        If None, then the transform will be applied to all the images withing the DataContainer.

    """

    def __init__(self, brightness_range=None, data_indices=None, p=0.5):
        super(ImageRandomBrightness, self).__init__(p=p, data_indices=data_indices)
        self._brightness_range = validate_numeric_range_parameter(brightness_range, (0, 0))

    def sample_transform(self):
        brightness_fact = random.uniform(self._brightness_range[0], self._brightness_range[1])
        lut = np.arange(0, 256) + brightness_fact
        lut = np.clip(lut, 0, 255).astype("uint8")
        self.state_dict = {'brightness_fact': brightness_fact, 'LUT': lut}

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        return cv2.LUT(img, self.state_dict['LUT'])


class ImageColorTransform(ImageTransform):
    """RGB to grayscale or grayscale to RGB image conversion.

    If converting from grayscale to RGB, then the gs channel is simply clonned.
    If converting from RGB to grayscale, then opencv is used.

    Parameters
    ----------
    mode : str
        Color conversion mode. If None, then no conversion happens and mode=none.
        If `mode == 'rgb2gs'` and the image is already grayscale,
        then nothing happens. If `mode == 'gs2rgb'` and the image is already RGB,
        then also nothing happens.
    data_indices : tuple or None
        Indices of the images within the data container to which this transform needs to be applied.
        Every element within the tuple must be integer numbers.
        If None, then the transform will be applied to all the images withing the DataContainer.

    """

    def __init__(self, mode=None, data_indices=None):
        super(ImageColorTransform, self).__init__(p=1, data_indices=data_indices)
        self._mode = validate_parameter(mode, allowed_color_conversions, 'none', heritable=False)

    @property
    def mode(self):
        return self._mode

    def sample_transform(self):
        pass

    @img_shape_checker
    def _apply_img(self, img: np.ndarray, settings: dict):
        if self._mode == 'none':
            return img
        elif self.mode == 'gs2rgb':
            if len(img.shape) != 3:
                raise ValueError
            if img.shape[-1] == 1:
                return np.dstack((img, img, img))
            elif img.shape[-1] == 3:
                return img
        elif self.mode == 'rgb2gs':
            if len(img.shape) != 3:
                raise ValueError
            if img.shape[-1] == 1:
                return img
            elif img.shape[-1] == 3:
                return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


class KeypointsJitter(DataDependentSamplingTransform):
    """
    Applies the jittering to the keypoints in X- and Y-durections

    Parameters
    ----------
    p : float
        Probability of applying the transform
    dx_range: None or int or tuple of int
        Jittering across X-axis. Valid range is (-1, 1)
    dy_range: None or int or tuple of int
        Jittering across Y-axis. Valid range is (-1, 1)

    """
    def __init__(self, p=0.5, dx_range=None, dy_range=None):
        super(KeypointsJitter, self).__init__(data_indices=None, p=p)

        self._dx_range = validate_numeric_range_parameter(dx_range, (0, 0), -1, 1)
        self._dy_range = validate_numeric_range_parameter(dy_range, (0, 0), -1, 1)

    def sample_transform(self):
        raise NotImplementedError

    def sample_transform_from_data(self, data: DataContainer):
        pass

    @img_shape_checker
    def _apply_img(self, img, settings: dict):
        return img

    def _apply_mask(self, mask, settings: dict):
        return mask

    def _apply_pts(self, pts: KeyPoints, settings: dict):
        pts_data = pts.data.copy()
        h = pts.H
        w = pts.W

        for j in range(pts.data.shape[0]):
            dx = int(random.uniform(self._dx_range[0], self._dx_range[1]) * w)
            dy = int(random.uniform(self._dy_range[0], self._dy_range[1]) * h)
            pts_data[j, 0] = min(pts_data[j, 0] + dx, w - 1)
            pts_data[j, 1] = min(pts_data[j, 1] + dy, h - 1)

        return KeyPoints(pts_data, h, w)

    def _apply_labels(self, labels, settings: dict):
        return labels
