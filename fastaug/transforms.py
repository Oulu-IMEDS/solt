from . import core, data
import cv2


class RandomFlip(core.BaseTransform):
    def __init__(self, p=0.5, axis=0):
        super(RandomFlip, self).__init__(p)
        self.params = None
        self.__axis = axis


    def sample_transform(self):
        # TODO: sample coordinates for remap, which will be used to fuse the transforms
        self.params = None

    @data.img_shape_checker
    def _apply_img(self, img):
        img = cv2.flip(img, self.__axis)
        return img

    def _apply_mask(self, mask):
        img = cv2.flip(mask, self.__axis)
        return img

    def _apply_labels(self, labels):
        return labels

    def _apply_pts(self, pts):
        if self.__axis == 0:
            pts.pts[:, 1] = pts.H - pts.pts[:, 1]
        if self.__axis == 1:
            pts.pts[:, 0] = pts.W - pts.pts[:, 0]


class RandomRotate(core.MatrixTransform):
    def __init__(self, p=0.5, interpolation='bilinear'):
        super(RandomRotate, self).__init__(p=p, interpolatio=interpolation)

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
