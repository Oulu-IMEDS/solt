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
    Random rotation around the center
    """
    def __init__(self, range, p=0.5, interpolation='bilinear'):
        super(RandomRotate, self).__init__(p=p, interpolatio=interpolation)
        self.__range = range

    def sample_transform(self):
        """
        Samples random rotation within specified range and saves it as an object state.

        """
        rot = np.random.uniform(self.__range[0], self.__range[1])
        M = np.array([np.cos(np.deg2rad(rot)), np.sin(np.deg2rad(rot)), 0,
                     -np.sin(np.deg2rad(rot)), np.cos(np.deg2rad(rot)), 0,
                     0, 0, 1
                     ]).reshape((3, 3)).astype(np.float32)

        self.params = {'rot': rot,
                       'transform_matrix': M}


class RandomScale(core.MatrixTransform):
    def __init__(self, p=0.5):
        super(RandomScale, self).__init__(p)

    def sample_transform(self):
        raise NotImplementedError

    @data.img_shape_checker
    def _apply_img(self, img):
        raise NotImplementedError

    def _apply_mask(self, mask):
        raise NotImplementedError

    def _apply_labels(self, labels):
        raise NotImplementedError

    def _apply_pts(self, pts):
        raise NotImplementedError


class RandomShear(core.MatrixTransform):
    def __init__(self, p=0.5):
        super(RandomShear, self).__init__(p)

    def sample_transform(self):
        raise NotImplementedError

    @data.img_shape_checker
    def _apply_img(self, img):
        raise NotImplementedError

    def _apply_mask(self, mask):
        raise NotImplementedError

    def _apply_labels(self, labels):
        raise NotImplementedError

    def _apply_pts(self, pts):
        raise NotImplementedError


class RandomCrop(core.BaseTransform):
    def __init__(self, p=0.5):
        super(RandomCrop, self).__init__(p)

    def sample_transform(self):
        raise NotImplementedError

    @data.img_shape_checker
    def _apply_img(self, img):
        raise NotImplementedError

    def _apply_mask(self, mask):
        raise NotImplementedError

    def _apply_labels(self, labels):
        raise NotImplementedError

    def _apply_pts(self, pts):
        raise NotImplementedError


class Pad(core.BaseTransform):
    def __init__(self, p=0.5):
        super(Pad, self).__init__(p)

    def sample_transform(self):
        raise NotImplementedError

    @data.img_shape_checker
    def _apply_img(self, img):
        raise NotImplementedError

    def _apply_mask(self, mask):
        raise NotImplementedError

    def _apply_labels(self, labels):
        raise NotImplementedError

    def _apply_pts(self, pts):
        raise NotImplementedError


class CenterCrop(core.BaseTransform):
    def __init__(self, p=0.5):
        super(CenterCrop, self).__init__(p)

    def sample_transform(self):
        raise NotImplementedError

    @data.img_shape_checker
    def _apply_img(self, img):
        raise NotImplementedError

    def _apply_mask(self, mask):
        raise NotImplementedError

    def _apply_labels(self, labels):
        raise NotImplementedError

    def _apply_pts(self, pts):
        raise NotImplementedError


class RandomPerspective(core.BaseTransform):
    def __init__(self, p=0.5):
        super(RandomPerspective, self).__init__(p)

    def sample_transform(self):
        raise NotImplementedError

    def _apply_img(self, img):
        raise NotImplementedError

    def _apply_mask(self, mask):
        raise NotImplementedError

    def _apply_labels(self, labels):
        raise NotImplementedError

    def _apply_pts(self, pts):
        raise NotImplementedError
