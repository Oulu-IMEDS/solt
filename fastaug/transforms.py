import numpy as np
import cv2

from .constants import allowed_paddings, allowed_crops, allowed_noise_types
from .data import img_shape_checker
from .data import KeyPoints, DataContainer
from .base_transforms import BaseTransform, MatrixTransform, PaddingPropertyHolder, DataDependentSamplingTransform
from .core import Pipeline


class RandomFlip(BaseTransform):
    """
    Performs a random flip of an image.

    """
    def __init__(self, p=0.5, axis=1):
        """
        Constructor.

        Parameters
        ----------
        p : float
            Probability of flip
        axis : int
            Axis of flip. Here, 0 stands for horizontal flipping, 1 stands for the vertical one.
        """
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
        interpolation : str or tuple or None.
            Interpolation type. Check the allowed interpolation types.
        padding : str or tuple or None.
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
        interpolation : str or tuple or None. or tuple or None.
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

        if isinstance(range_x, (int, float)):
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
        interpolation : str or tuple or None.
            Interpolation type. Check the allowed interpolation types. 
            one indicates default behavior - bilinear mode.
        p : float
            Probability of using this transform
        """
        super(RandomScale, self).__init__(interpolation=interpolation, padding=None, p=p)

        if isinstance(range_x, (int, float)):
            range_x = (-range_x, range_x)

        if isinstance(range_y, (int, float)):
            range_y = (-range_y, range_y)

        if range_x is not None:
            assert (range_x[0] > 0 and range_x[0] > 0)
        if range_y is not None:
            assert (range_y[0] > 0 and range_y[0] > 0)

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
    This transform does not change the original frame where the transform is applied.

    """
    def __init__(self, range_x=None, range_y=None,  interpolation='bilinear', padding='z', p=0.5):
        super(RandomTranslate, self).__init__(interpolation=interpolation, padding=padding, p=p)
        if isinstance(range_x, (int, float)):
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

        M = np.array([1, 0, tx,
                      0, 1, ty,
                      0, 0, 1]).reshape((3, 3)).astype(np.float32)

        self.state_dict = {'scale_x': tx, 'scale_y': ty, 'transform_matrix': M}


class RandomProjection(MatrixTransform):
    """
    Generates random Perspective transform. Takes a set of affine transforms and generates a projective
    transform according to the eq. 2.13 from A. Zisserman's book: Multiple View Geometry in Computer Vision.

    """
    def __init__(self, affine_transforms=None, v_range=None, p=0.5):
        """
        Constructor.

        Parameters
        ----------
        affine_transforms : Pipeline or None
            Pipeline object, which has a parameterized Affine Transform. If it is None, then empty pipeline is created.
        v_range : tuple or None.
            Projective parameters range. If None, then v_range = (0, 0)
        p : float
            Probability of using this transform
        """
        super(RandomProjection, self).__init__(p=p)

        if affine_transforms is None:
            affine_transforms = Pipeline()

        if v_range is None:
            v_range = (0, 0)
        assert isinstance(affine_transforms, Pipeline)
        for trf in affine_transforms.transforms:
            assert isinstance(trf, MatrixTransform)

        assert isinstance(v_range, tuple)
        for limit in v_range:
            assert isinstance(limit, (int, float))

        self.__affine_transforms = affine_transforms
        self.__vrange = v_range  # projection components.

    def sample_transform(self):
        trf = Pipeline.optimize_stack(self.__affine_transforms.transforms)[0]
        M = trf.state_dict['transform_matrix'].copy()
        M[-1, 0] = np.random.uniform(self.__vrange[0], self.__vrange[1])
        M[-1, 1] = np.random.uniform(self.__vrange[0], self.__vrange[1])
        self.state_dict['transform_matrix'] = M


class PadTransform(BaseTransform, PaddingPropertyHolder):
    def __init__(self, pad_to, padding=None):
        """
        Transformation, which pads the input to a given size

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
        BaseTransform.__init__(self, p=1)
        PaddingPropertyHolder.__init__(self, padding)
        assert isinstance(pad_to, tuple) or isinstance(pad_to, int)
        if isinstance(pad_to, int):
            pad_to = tuple(pad_to, pad_to)

        self._pad_to = pad_to

    def sample_transform(self):
        pass

    def _apply_img_or_mask(self, img):
        h = img.shape[0]
        w = img.shape[1]

        pad_w = (self._pad_to[0] - w) // 2
        pad_h = (self._pad_to[1] - h) // 2
        assert pad_h >= 0
        assert pad_w >= 0

        padding = allowed_paddings[self.padding[0]]
        return cv2.copyMakeBorder(img, pad_h, pad_h+(self._pad_to[1] - h) % 2, pad_w, pad_w + (self._pad_to[0] - w) % 2, padding, value=0)

    @img_shape_checker
    def _apply_img(self, img):
        return self._apply_img_or_mask(img)

    def _apply_mask(self, mask):
        return self._apply_img_or_mask(mask)

    def _apply_labels(self, labels):
        return labels

    def _apply_pts(self, pts):
        assert self.padding[0] == 'z'
        pts_data = pts.data.copy()

        pad_w = (self._pad_to[0] - pts.H) // 2
        pad_h = (self._pad_to[1] - pts.W) // 2

        pts_data[:, 0] += pad_w
        pts_data[:, 1] += pad_h

        return KeyPoints(pts_data, self._pad_to[1], self._pad_to[0])


class CropTransform(DataDependentSamplingTransform):
    def __init__(self, crop_size, crop_mode='c'):
        """
        Constructor

        Parameters
        ----------
        crop_size : tuple or int
            Size of the crop (W_new, H_new). If int, then a square crop will be made.
        crop_mode : str
            Crop mode. Can be either 'c' - center or 'r' - random.

        """
        super(CropTransform, self).__init__(p=1, data_indices=None)

        assert isinstance(crop_size, int) or isinstance(crop_size, tuple)
        assert crop_mode in allowed_crops

        if isinstance(crop_size, tuple):
            assert isinstance(crop_size[0], int)

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        self._crop_size = crop_size
        self._crop_mode = crop_mode

    @property
    def crop_mode(self):
        return self._crop_mode

    @crop_mode.setter
    def crop_mode(self, value):
        assert value in allowed_crops
        self._crop_mode = value

    @property
    def crop_size(self):
        return self._crop_size

    @crop_size.setter
    def crop_size(self, value):
        assert isinstance(value, int) or isinstance(value, tuple)
        if isinstance(value, tuple):
            assert isinstance(value, int)
        self._crop_size = value

    def sample_transform(self):
        raise NotImplementedError

    def sample_transform_from_data(self, data):
        # calling the superclass method to ensure that everything is right with the coordinates
        DataDependentSamplingTransform.sample_transform_from_data(self, data)

        for obj, t in data:
            if t == 'M' or t == 'I':
                h = obj.shape[0]
                w = obj.shape[1]
            elif t == 'P':
                h = obj.H
                w = obj.W
            else:
                continue

        assert self.crop_size[0] < w
        assert self.crop_size[1] < h

        if self.crop_mode == 'c':
            x = w // 2 - self.crop_size[0] // 2
            y = h // 2 - self.crop_size[1] // 2
        elif self.crop_mode == 'r':
            x = np.random.randint(0, w - self.crop_size[0])
            y = np.random.randint(0, h - self.crop_size[1])
        else:
            raise NotImplementedError

        self.state_dict = {'x': x, 'y':y}

    def _crop_img_or_mask(self, img):
        assert 'x' in self.state_dict
        assert 'y' in self.state_dict
        x, y = self.state_dict['x'], self.state_dict['y']
        return img[y:y+self.crop_size[1], x:x+self.crop_size[0]]

    @img_shape_checker
    def _apply_img(self, img):
        return self._crop_img_or_mask(img)

    def _apply_mask(self, mask):
        return self._crop_img_or_mask(mask)

    def _apply_labels(self, labels):
        return labels

    def _apply_pts(self, pts):
        pts_data = pts.data.copy()
        assert 'x' in self.state_dict
        assert 'y' in self.state_dict
        x, y = self.state_dict['x'], self.state_dict['y']

        pts_data[:, 0] -= x
        pts_data[:, 1] -= y

        return KeyPoints(pts_data, self.crop_size[1], self.crop_size[0])


class ImageAdditiveGaussianNoise(DataDependentSamplingTransform):
    """
    Adds noise to an image. Other types of data than the image are ignored.

    """
    def __init__(self, p=0.5, gain_range=0.1, data_indices=None):
        """
        Constructor.

        Parameters
        ----------
        p : float
            Probability of applying this transfor,
        gain_range : tuple or float
            Gain of the noise. Final image is created as (1-gain)*img + gain*noise.
            If float, then gain_range = (0, gain_range).
        data_indices : tuple or None
            Indices of the images within the data container to which this transform needs to be applied.
            Every element within the tuple must be integer numebers.
            If None, then the transform will be applied to all the images withing the DataContainer.
        """
        super(ImageAdditiveGaussianNoise, self).__init__(p=p, data_indices=data_indices)
        if isinstance(gain_range, float):
            gain_range = (0, gain_range)

        assert gain_range[0] >= 0
        assert gain_range[1] <= 1

        self._gain_range = gain_range

    def sample_transform(self):
        raise NotImplementedError

    def sample_transform_from_data(self, data: DataContainer):
        DataDependentSamplingTransform.sample_transform_from_data(self, data)
        gain = np.random.uniform(self._gain_range[0], self._gain_range[1])
        h = None
        w = None
        c = None
        for obj, t in data:
            if t == 'I':
                h = obj.shape[0]
                w = obj.shape[1]
                c = obj.shape[2]
                break

        assert w is not None
        assert h is not None
        assert c is not None

        noise_img = np.random.randn(h, w, c)

        noise_img -= noise_img.min()
        noise_img /= noise_img.max()
        noise_img *= 255
        noise_img = noise_img.astype(obj.dtype)

        self.state_dict = {'noise': noise_img, 'gain': gain}

    @img_shape_checker
    def _apply_img(self, img):
        return cv2.addWeighted(img, (1-self.state_dict['gain']), self.state_dict['noise'], self.state_dict['gain'], 0)

    def _apply_mask(self, mask):
        return mask

    def _apply_labels(self, labels):
        return labels

    def _apply_pts(self, pts):
        return pts


class ImageSaltAndPepper(DataDependentSamplingTransform):
    """
    Adds salt and pepper noise to an image. Other types of data than the image are ignored.

    """
    def __init__(self, p=0.5, gain_range=0.1, salt_p=0.5, data_indices=None):
        """
        Constructor.

        Parameters
        ----------
        p : float
            Probability of applying this transfor,
        gain_range : tuple or float
            Gain of the noise. Indicates percentage of indices, which will be changed.
            If float, then gain_range = (0, gain_range).
        salt_p : float or tuple
            Percentage of salt. Percentage of pepper is 1-salt_p. If tuple, then salt_p is chosen uniformly from the
            given range.
        data_indices : tuple or None
            Indices of the images within the data container to which this transform needs to be applied.
            Every element within the tuple must be integer numebers.
            If None, then the transform will be applied to all the images withing the DataContainer.
        """
        super(ImageSaltAndPepper, self).__init__(p=p, data_indices=data_indices)

        assert isinstance(gain_range, float) or isinstance(gain_range, tuple)
        assert isinstance(salt_p, float) or isinstance(salt_p, tuple)

        if isinstance(gain_range, float):
            gain_range = (0, gain_range)

        if isinstance(salt_p, float):
            assert 0 <= salt_p <= 1
            salt_p = (salt_p, salt_p)

        assert len(salt_p) == 2
        assert salt_p[0] <= salt_p[1]
        assert 0 <= salt_p[0] <= 1
        assert 0 <= salt_p[1] <= 1

        assert gain_range[0] >= 0
        assert gain_range[1] <= 1

        self._gain_range = gain_range
        self._salt_p = salt_p

    def sample_transform(self):
        raise NotImplementedError

    def sample_transform_from_data(self, data: DataContainer):
        DataDependentSamplingTransform.sample_transform_from_data(self, data)
        gain = np.random.uniform(self._gain_range[0], self._gain_range[1])
        salt_p = np.random.uniform(self._salt_p[0], self._salt_p[1])
        h = None
        w = None
        c = None
        for obj, t in data:
            if t == 'I':
                h = obj.shape[0]
                w = obj.shape[1]
                c = obj.shape[2]
                break

        assert w is not None
        assert h is not None
        assert c is not None

        sp = np.random.rand(h, w) <= gain
        salt = sp.copy() * 1.
        pepper = sp.copy() * 1.
        salt_mask = (np.random.rand(sp.sum()) <= salt_p)
        pepper_mask = 1 - salt_mask
        salt[np.where(salt)] *= salt_mask
        pepper[np.where(pepper)] *= pepper_mask

        self.state_dict = {'salt': salt, 'pepper': pepper}

    @img_shape_checker
    def _apply_img(self, img):
        img = img.copy()
        # TODO: uint16 support
        img[np.where(self.state_dict['salt'])] = 255
        img[np.where(self.state_dict['pepper'])] = 0
        return img

    def _apply_mask(self, mask):
        return mask

    def _apply_labels(self, labels):
        return labels

    def _apply_pts(self, pts):
        return pts


class ImageGammaCorrection(BaseTransform):
    """
    Transform applies random gamma correction

    """
    def __init__(self, p=0.5, gamma_range=0.1, data_indices=None):
        """
        Constructor.

        Parameters
        ----------
        p : float
            Probability of applying this transfor,
        gamma_range : tuple or float
            Gain of the noise. Indicates percentage of indices, which will be changed.
            If float, then gain_range = (1-gamma_range, 1+gamma_range).
        data_indices : tuple or None
            Indices of the images within the data container to which this transform needs to be applied.
            Every element within the tuple must be integer numebers.
            If None, then the transform will be applied to all the images withing the DataContainer.
        """
        super(ImageGammaCorrection, self).__init__(p=p, data_indices=data_indices)

        assert isinstance(gamma_range, float) or isinstance(gamma_range, tuple)

        if isinstance(gamma_range, float):
            gamma_range = (1-gamma_range, 1+gamma_range)

        assert gamma_range[0] >= 0

        self._gamma_range = gamma_range

    def sample_transform(self):
        gamma = np.random.uniform(self._gamma_range[0], self._gamma_range[1])
        inv_gamma = 1.0 / gamma
        lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

        self.state_dict = {'gamma': gamma, 'LUT':lut}

    @img_shape_checker
    def _apply_img(self, img):
        return cv2.LUT(img, self.state_dict['LUT'])

    def _apply_mask(self, mask):
        return mask

    def _apply_labels(self, labels):
        return labels

    def _apply_pts(self, pts):
        return pts






