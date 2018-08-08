from . import core, data
import cv2
import numpy as np


class RandomFlip(core.BaseTransform):
    def __init__(self, p=0.5, axis=0):
        super(RandomFlip, self).__init__(p)
        self.params = None
        self.__axis = axis

    def sample_transform(self):
        # TODO: sample coordinates for remap, which will be used to fuse the transforms
        pass

    @data.img_shape_checker
    def _apply_img(self, img):
        img = cv2.flip(img, self.__axis)
        return img

    def _apply_mask(self, mask):
        mask_new = cv2.flip(mask, self.__axis)
        return mask_new

    def _apply_labels(self, labels):
        return labels

    def _apply_pts(self, pts):
        if self.__axis == 1:
            pts.data[:, 1] = pts.H - 1 - pts.data[:, 1]
        if self.__axis == 0:
            pts.data[:, 0] = pts.W - 1 - pts.data[:, 0]

        return pts


class RandomRotate(core.MatrixTransform):
    """
    Random rotation around the center.
    """
    def __init__(self, rotation_range, padding='zeros', p=0.5):
        """
        Constructor.

        Parameters
        ----------
        rotation_range : rotation range
        p : probability of using this transform
        """
        super(RandomRotate, self).__init__(p=p,  padding=padding)

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

        self.params = {'rot': rot,
                       'transform_matrix': M}


class RandomShear(core.MatrixTransform):
    """
    Random shear around the center.

    """
    def __init__(self, range_x, range_y, p=0.5):
        """
        Constructor.

        Parameters
        ----------
        range_x : shearing range along X-axis
        range_y : shearing range along Y-axis
        p : probability of using the transform
        """
        super(RandomShear, self).__init__(p)
        self.__range_x = range_x
        self.__range_y = range_y

    def sample_transform(self):
        shear_x = np.random.uniform(self.__range_x[0], self.__range_x[1])
        shear_y = np.random.uniform(self.__range_y[0], self.__range_y[1])

        M = np.array([1, shear_y, 0,
                     shear_x, 1, 0,
                     0, 0, 1]).reshape((3, 3)).astype(np.float32)

        self.params = {'shear_x': shear_x,
                       'shear_y': shear_y,
                       'transform_matrix': M}


class RandomScale(core.MatrixTransform):
    """
    Random scale transform.

    """
    def __init__(self, range_x, range_y, p=0.5):
        super(RandomScale, self).__init__(p)
        self.__range_x = range_x
        self.__range_y = range_y

    def sample_transform(self):
        scale_x = np.random.uniform(self.__range_x[0], self.__range_x[1])
        scale_y = np.random.uniform(self.__range_y[0], self.__range_y[1])

        M = np.array([scale_x, 0, 0,
                      0, scale_y, 0,
                      0, 0, 1]).reshape((3, 3)).astype(np.float32)

        self.params = {'scale_x': scale_x,
                       'scale_y': scale_y,
                       'transform_matrix': M}


class RandomCrop(core.BaseTransform):
    def __init__(self, crop_size):
        super(RandomCrop, self).__init__(p=1)
        self.crop_size = crop_size

    def sample_transform(self):
        raise NotImplementedError

    @data.img_shape_checker
    def _apply_img(self, img):
        raise NotImplementedError

    def _apply_mask(self, mask):
        raise NotImplementedError

    def _apply_labels(self, labels):
        return labels

    def _apply_pts(self, pts):
        raise NotImplementedError


class Pad(core.BaseTransform):
    def __init__(self, pad_to):
        super(Pad, self).__init__(p=1)
        self.__pad_to = pad_to

    def sample_transform(self):
        pass

    @data.img_shape_checker
    def _apply_img(self, img):
        raise NotImplementedError

    def _apply_mask(self, mask):
        raise NotImplementedError

    def _apply_labels(self, labels):
        return labels

    def _apply_pts(self, pts):
        raise NotImplementedError


class CenterCrop(core.BaseTransform):
    def __init__(self, crop_size):
        super(CenterCrop, self).__init__(p=1)
        self.crop_size = crop_size

    def sample_transform(self):
        pass

    @data.img_shape_checker
    def _apply_img(self, img):
        raise NotImplementedError

    def _apply_mask(self, mask):
        raise NotImplementedError

    def _apply_labels(self, labels):
        raise NotImplementedError

    def _apply_pts(self, pts):
        raise NotImplementedError


class RandomPerspective(core.MatrixTransform):
    def __init__(self, tilt_range, p=0.5):
        super(RandomPerspective, self).__init__(p)
        self.__tilt_range = tilt_range

    def sample_transform(self):
        raise NotImplementedError

