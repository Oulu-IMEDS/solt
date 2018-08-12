import numpy as np
import cv2


from .data import img_shape_checker
from .data import KeyPoints
from .base_transforms import BaseTransform, MatrixTransform


class RandomFlip(BaseTransform):
    """
    Performs a random flip of an image.

    """
    def __init__(self, p=0.5, axis=1):
        super(RandomFlip, self).__init__(p=p)
        self.__axis = axis

    def sample_transform(self):
        pass

    @img_shape_checker
    def _apply_img(self, img):
        img = cv2.flip(img, self.__axis)
        return img

    def _apply_mask(self, mask):
        mask_new = cv2.flip(mask, self.__axis)
        return mask_new

    def _apply_labels(self, labels):
        return labels

    def _apply_pts(self, pts):
        # We should guarantee that we do not change the original data.
        pts_data = pts.data.copy()
        if self.__axis == 0:
            pts_data[:, 1] = pts.H - 1 - pts_data[:, 1]
        if self.__axis == 1:
            pts_data[:, 0] = pts.W - 1 - pts_data[:, 0]
        return KeyPoints(pts=pts_data, H=pts.H, W=pts.W)


class RandomRotate(MatrixTransform):
    """
    Random rotation around the center.

    """
    def __init__(self, rotation_range=None, interpolation='bilinear', padding='z', p=0.5):
        """
        Constructor.

        Parameters
        ----------
        rotation_range : tuple or float or None
            Range of rotation.
            If float, then (-rotation_range, rotation_range) will be used for transformation sampling.
            if None, then rotation_range=(0,0).
        interpolation : str
            Interpolation type. Check the allowed interpolation types.
        padding : str
            Padding mode. Check the allowed padding modes.
        p : float
            Probability of using this transform
        """
        super(RandomRotate, self).__init__(interpolation=interpolation, padding=padding, p=p)
        if rotation_range is None:
            rotation_range = (0, 0)

        if isinstance(rotation_range, (int, float)):
            rotation_range = (-rotation_range, rotation_range)

        self.__range = rotation_range

    def sample_transform(self):
        """
        Samples random rotation within specified range and saves it as an object state.

        """
        rot = np.random.uniform(self.__range[0], self.__range[1])
        M = np.array([np.cos(np.deg2rad(rot)), -np.sin(np.deg2rad(rot)), 0,
                     np.sin(np.deg2rad(rot)), np.cos(np.deg2rad(rot)), 0,
                     0, 0, 1
                     ]).reshape((3, 3)).astype(np.float32)

        self.state_dict = {'rot': rot, 'transform_matrix': M}


class RandomShear(MatrixTransform):
    """
    Random shear around the center.

    """
    def __init__(self, range_x=None, range_y=None, interpolation='bilinear', padding='z', p=0.5):
        """
        Constructor.

        Parameters
        ----------
        range_x : tuple or float
            Shearing range along X-axis.
            If float, then (-range_x, range_x) will be used.
            If None, then range_x=(0, 0)
        range_y : tuple or float or None
            Shearing range along Y-axis. If float, then (-range_y, range_y) will be used.
            If None, then range_y=(0, 0)
        interpolation : str
            Interpolation type. Check the allowed interpolation types.
        padding : str
            Padding mode. Check the allowed padding modes.
        p : float
            Probability of using this transform
        """
        super(RandomShear, self).__init__(p=p, padding=padding, interpolation=interpolation)
        if range_x is None:
            range_x = (0, 0)
        if range_y is None:
            range_y = (0, 0)

        if isinstance(range_y, (int, float)):
            range_x = (-range_x, range_x)

        if isinstance(range_y, (int, float)):
            range_y = (-range_y, range_y)

        self.__range_x = range_x
        self.__range_y = range_y

    def sample_transform(self):
        shear_x = np.random.uniform(self.__range_x[0], self.__range_x[1])
        shear_y = np.random.uniform(self.__range_y[0], self.__range_y[1])

        M = np.array([1, shear_x, 0,
                     shear_y, 1, 0,
                     0, 0, 1]).reshape((3, 3)).astype(np.float32)

        self.state_dict = {'shear_x': shear_x, 'shear_y': shear_y, 'transform_matrix': M}


class RandomScale(MatrixTransform):
    """
    Random scale transform.

    """
    def __init__(self, range_x=None, range_y=None, same=True, interpolation='bilinear', p=0.5):
        """
        Constructor.

        Parameters
        ----------
        range_x : tuple or float or None
            Scaling range along X-axis.
            If float, then (-range_x, range_x) will be used.
            If None, then range_x =(1,1) by default.
        range_y : tuple or float
            Scaling range along Y-axis.
            If float, then (-range_y, range_y) will be used.
            If None, then range_y=(1,1) by default.
        same: bool
            Indicates whether to use the same scaling factor for width and height.
        interpolation : str
            Interpolation type. Check the allowed interpolation types.
        p : float
            Probability of using this transform
        """
        super(RandomScale, self).__init__(interpolation=interpolation, padding=None, p=p)

        if isinstance(range_y, (int, float)):
            range_x = (-range_x, range_x)

        if isinstance(range_y, (int, float)):
            range_y = (-range_y, range_y)

        self.__same = same
        self.__range_x = range_x
        self.__range_y = range_y

    def sample_transform(self):
        if self.__range_x is None:
            scale_x = 1
        else:
            scale_x = np.random.uniform(self.__range_x[0], self.__range_x[1])

        if self.__range_y is None:
            scale_y = 1
        else:
            scale_y = np.random.uniform(self.__range_y[0], self.__range_y[1])

        if self.__same:
            if self.__range_x is None:
                scale_x = scale_y
            else:
                scale_y = scale_x

        M = np.array([scale_x, 0, 0,
                      0, scale_y, 0,
                      0, 0, 1]).reshape((3, 3)).astype(np.float32)

        self.state_dict = {'scale_x': scale_x, 'scale_y': scale_y, 'transform_matrix': M}


class RandomTranslate(MatrixTransform):
    """
    Random Translate transform.

    """
    def __init__(self, range_x=None, range_y=None,  interpolation='bilinear', padding='z', p=0.5):
        super(RandomTranslate, self).__init__(interpolation=interpolation, padding=padding, p=p)
        if isinstance(range_y, (int, float)):
            range_x = (-range_x, range_x)

        if isinstance(range_y, (int, float)):
            range_y = (-range_y, range_y)

        self.__range_x = range_x
        self.__range_y = range_y

    def sample_transform(self):
        if self.__range_x is None:
            tx = 0
        else:
            tx = np.random.uniform(self.__range_x[0], self.__range_x[1])

        if self.__range_y is None:
            ty = 0
        else:
            ty = np.random.uniform(self.__range_y[0], self.__range_y[1])

        M = np.array([0, 0, tx,
                      0, 0, ty,
                      0, 0, 1]).reshape((3, 3)).astype(np.float32)

        self.state_dict = {'scale_x': tx, 'scale_y': ty, 'transform_matrix': M}


class RandomCrop(BaseTransform):
    def __init__(self, crop_size):
        super(RandomCrop, self).__init__(p=1)
        self.__crop_size = crop_size

    def sample_transform(self):
        raise NotImplementedError

    @img_shape_checker
    def _apply_img(self, img):
        assert self.__crop_size[0] < img.shape[1]
        assert self.__crop_size[1] < img.shape[0]
        raise NotImplementedError

    def _apply_mask(self, mask):
        assert self.__crop_size[0] < mask.shape[1]
        assert self.__crop_size[1] < mask.shape[0]
        raise NotImplementedError

    def _apply_labels(self, labels):
        return labels

    def _apply_pts(self, pts):
        assert self.__crop_size[0] < pts.W
        assert self.__crop_size[1] < pts.H
        raise NotImplementedError


class RandomProjective(MatrixTransform):
    """
    Generates random Perspective transform. Takes a set of affine transforms and generates a projective
    transform according to the eq. 2.13 from A. Zisserman's book: Multiple View Geometry in Computer Vision.

    """
    def __init__(self, affine_transforms, v_range, p=0.5):
        super(RandomProjective, self).__init__(p=p)
        self.affine_transforms = affine_transforms

    def sample_transform(self):
        raise NotImplementedError


class Pad(BaseTransform):
    def __init__(self, pad_to):
        super(Pad, self).__init__(p=1)
        self.__pad_to = pad_to

    def sample_transform(self):
        pass

    @img_shape_checker
    def _apply_img(self, img):
        raise NotImplementedError

    def _apply_mask(self, mask):
        raise NotImplementedError

    def _apply_labels(self, labels):
        return labels

    def _apply_pts(self, pts):
        raise NotImplementedError


class CenterCrop(BaseTransform):
    def __init__(self, crop_size):
        super(CenterCrop, self).__init__(p=1)
        self.crop_size = crop_size

    def sample_transform(self):
        pass

    @img_shape_checker
    def _apply_img(self, img):
        raise NotImplementedError

    def _apply_mask(self, mask):
        raise NotImplementedError

    def _apply_labels(self, labels):
        raise NotImplementedError

    def _apply_pts(self, pts):
        raise NotImplementedError

