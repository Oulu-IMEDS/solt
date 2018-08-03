from . import core
import cv2

class RandomRotate(core.MatrixTransform):
    def __init__(self, p=0.5, interpolation='bilinear'):
        super(RandomRotate, self).__init__(p=p, interpolatio=interpolation)

class RandomScale(core.MatrixTransform):
    def __init__(self, p=0.5):
        super(RandomScale, self).__init__(p)


class RandomShear(core.MatrixTransform):
    def __init__(self, p=0.5):
        super(RandomShear, self).__init__(p)


class RandomFlip(core.BasicTransform):
    def __init__(self, p=0.5, axis=0):
        super(RandomFlip, self).__init__(p)
        self.params = None
        self.axis = axis

    def sample_transform(self):
        # TODO: sample coordinates for remap
        # For now it is just a placeholder
        self.params = None

    def _apply_img(self, img):
        # TODO: use remap in the next version
        img = cv2.flip(img, self.axis)
        return img

    def _apply_mask(self, mask):
        # TODO: use remap in the next version
        img = cv2.flip(mask, self.axis)
        return img

    def _apply_labels(self, labels):
        # TODO: use remap in the next version
        return labels

    def _apply_pts(self, pts, h, w):
        raise NotImplementedError


class RandomCrop(core.BasicTransform):
    def __init__(self, p=0.5):
        super(RandomCrop, self).__init__(p)


class Pad(core.BasicTransform):
    def __init__(self, p=0.5):
        super(Pad, self).__init__(p)


class CenterCrop(core.BasicTransform):
    def __init__(self, p=0.5):
        super(CenterCrop, self).__init__(p)


class RandomHomography(core.BasicTransform):
    def __init__(self, p=0.5):
        super(RandomHomography, self).__init__(p)