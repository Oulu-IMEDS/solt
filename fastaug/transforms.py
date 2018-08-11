import numpy as np
from abc import ABCMeta, abstractmethod
import cv2

from .data import img_shape_checker
from .data import DataContainer, KeyPoints
from .constants import allowed_interpolations, allowed_paddings


class BaseTransform(metaclass=ABCMeta):
    """
    Transformation abstract class.

    """
    def __init__(self, p=0.5):
        """
        Constructor.

        Parameters
        ----------
        p : probability of executing this transform

        """
        self.p = p
        self.use = False  # Initially we do not use the transform
        self.params = None

    def use_transform(self):
        """
        Method to randomly determine whether to use this transform.

        Returns
        -------
        out : bool
            Boolean flag. True if the transform is used.
        """
        if np.random.rand() < self.p:
            self.use = True
            return True

        self.use = False
        return False

    @abstractmethod
    def sample_transform(self):
        """
        Abstract method. Must be implemented in the child classes

        Returns
        -------
        None

        """
        pass

    def apply(self, data):
        """
        Applies transformation to a DataContainer items depending on the type.

        Parameters
        ----------
        data : DataContainer
            Data to be augmented

        Returns
        -------
        out : DataContainer
            Result

        """
        result = []
        types = []
        for i, (item, t) in enumerate(data):
            if t == 'I':  # Image
                tmp_item = self._apply_img(item)
            elif t == 'M':  # Mask
                tmp_item = self._apply_mask(item)
            elif t == 'P':  # Points
                tmp_item = self._apply_pts(item)
            else:  # Labels
                tmp_item = self._apply_labels(item)

            types.append(t)
            result.append(tmp_item)

        return DataContainer(data=tuple(result), fmt=''.join(types))

    def __call__(self, data):
        """
        Applies the transform to a DataContainer

        Parameters
        ----------
        data : DataContainer
            Data to be augmented

        Returns
        -------
        out : DataContainer
            Result

        """
        self.use_transform()
        self.sample_transform()
        if self.use:
            return self.apply(data)
        else:
            return data

    @abstractmethod
    def _apply_img(self, img):
        """
        Abstract method, which determines the transform's behaviour when it is applied to images HxWxC.

        Parameters
        ----------
        img : ndarray
            Image to be augmented

        Returns
        -------
        out : ndarray

        """
        pass

    @abstractmethod
    def _apply_mask(self, mask):
        """
        Abstract method, which determines the transform's behaviour when it is applied to masks HxW.

        Parameters
        ----------
        mask : ndarray
            Mask to be augmented

        Returns
        -------
        out : ndarray
            Result

        """
        pass

    @abstractmethod
    def _apply_labels(self, labels):
        """
        Abstract method, which determines the transform's behaviour when it is applied to labels (e.g. label smoothing)

        Parameters
        ----------
        labels : ndarray
            Array of labels.

        Returns
        -------
        out : ndarray
            Result

        """
        pass

    @abstractmethod
    def _apply_pts(self, pts):
        """
        Abstract method, which determines the transform's behaviour when it is applied to keypoints.

        Parameters
        ----------
        pts : KeyPoints
            Keypoints object

        Returns
        -------
        out : KeyPoints
            Result

        """
        pass


class MatrixTransform(BaseTransform):
    """
    Matrix Transform abstract class. (Affine and Homography).
    Does all the transforms around the image /  center.

    """
    def __init__(self, interpolation='bilinear', padding='z', p=0.5):
        assert padding in allowed_paddings
        # TODO: interpolation for each item within data container
        assert interpolation in allowed_interpolations
        super(MatrixTransform, self).__init__(p=p)
        self.padding = padding
        self.interpolation = interpolation
        self.params = {'transform_matrix': np.eye(3)}

    def fuse_with(self, trf):
        """
        Takes a transform an performs a matrix fusion. This is useful to optimize the computations

        Parameters
        ----------
        trf : MatrixTransform

        """
        assert self.params is not None
        assert trf.params is not None

        if not isinstance(trf, RandomScale):
            self.padding = trf.padding
        self.interpolation = trf.interpolation

        self.params['transform_matrix'] = self.params['transform_matrix'] @ trf.params['transform_matrix']

    @abstractmethod
    def sample_transform(self):
        """
        Abstract method. Must be implemented in the child classes

        Returns
        -------
        None

        """
        pass

    @staticmethod
    def correct_for_frame_change(M, W, H):
        """
        Method takes a matrix transform, and modifies its origin.

        Parameters
        ----------
        M : ndarray
            Transform (3x3) matrix
        W : int
            Width of the coordinate frame
        H : int
            Height of the coordinate frame
        Returns
        -------
        out : ndarray
            Modified Transform matrix

        """
        # First we correct the transformation so that it is performed around the origin
        origin = [(W-1) // 2, (H-1) // 2]
        T_origin = np.array([1, 0, -origin[0],
                             0, 1, -origin[1],
                             0, 0, 1]).reshape((3, 3))

        T_origin_back = np.array([1, 0, origin[0],
                                  0, 1, origin[1],
                                  0, 0, 1]).reshape((3, 3))

        # TODO: Check whether translation works and use this matrix if possible
        T_initial = np.array([1, 0, M[0, 2],
                             0, 1, M[1, 2],
                             0, 0, 1]).reshape((3, 3))

        M = T_origin_back @ M @ T_origin

        # Now, if we think of scaling, rotation and translation, the image gets increased when we
        # apply any transform.

        # This is needed to recalculate the size of the image after the transformation.
        # The core idea is to transform the coordinate grid
        # left top, left bottom, right bottom, right top
        coord_frame = np.array([[0, 0, 1], [0, H, 1], [W, H, 1], [W, 0, 1]])
        new_frame = np.dot(M, coord_frame.T).T
        new_frame[:, 0] /= new_frame[:, -1]
        new_frame[:, 1] /= new_frame[:, -1]
        new_frame = new_frame[:, :-1]
        # Computing the new coordinates

        # If during the transform, we obtained negativa coordinates, we have to move to the origin
        if np.any(new_frame[:, 0] < 0):
            new_frame[:, 0] += abs(new_frame[:, 0].min())
        if np.any(new_frame[:, 1] < 0):
            new_frame[:, 1] += abs(new_frame[:, 1].min())
        # In case of scaling the coordinate_frame, we need to move back to the origin
        new_frame[:, 0] -= new_frame[:, 0].min()
        new_frame[:, 1] -= new_frame[:, 1].min()

        W_new = int(np.round(new_frame[:, 0].max()))
        H_new = int(np.round(new_frame[:, 1].max()))

        M[0, -1] += W_new//2-origin[0]
        M[1, -1] += H_new//2-origin[1]

        return M, W_new, H_new

    @img_shape_checker
    def _apply_img(self, img):
        """
        Applies a matrix transform to an image.

        """
        M = self.params['transform_matrix']
        M, W_new, H_new = MatrixTransform.correct_for_frame_change(M, img.shape[1], img.shape[0])

        interp = allowed_interpolations[self.interpolation]
        if self.padding == 'z':
            return cv2.warpPerspective(img, M , (W_new, H_new), interp,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            return cv2.warpPerspective(img, M, (W_new, H_new), interp,
                                       borderMode=cv2.BORDER_REFLECT)

    def _apply_mask(self, mask):
        """
        Abstract method, which determines the transform's behaviour when it is applied to masks HxW.

        Parameters
        ----------
        mask : ndarray
            Mask to be augmented

        Returns
        -------
        out : ndarray
            Result

        """
        # X, Y coordinates
        M = self.params['transform_matrix']
        M, W_new, H_new = MatrixTransform.correct_for_frame_change(M, mask.shape[1], mask.shape[0])
        interp = allowed_interpolations[self.interpolation]
        if self.padding == 'z':
            return cv2.warpPerspective(mask, M , (W_new, H_new), interp,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            return cv2.warpPerspective(mask, M, (W_new, H_new), interp,
                                       borderMode=cv2.BORDER_REFLECT)

    def _apply_labels(self, labels):
        """
        Transform application to labels. Simply returns them.

        Parameters
        ----------
        labels : ndarray
            Array of labels.

        Returns
        -------
        out : ndarray
            Result

        """
        return labels

    def _apply_pts(self, pts):
        """
        Abstract method, which determines the transform's behaviour when it is applied to keypoints.

        Parameters
        ----------
        pts : KeyPoints
            Keypoints object

        Returns
        -------
        out : KeyPoints
            Result

        """
        if self.padding == 'r':
            raise ValueError('Cannot apply transform to keypoints with reflective padding!')

        pts_data = pts.data
        M = self.params['transform_matrix']
        M, W_new, H_new = MatrixTransform.correct_for_frame_change(M, pts.W, pts.H)

        pts_data = np.hstack((pts_data, np.ones((pts_data.shape[0], 1))))
        pts_data = np.dot(M, pts_data.T).T

        pts_data[:, 0] /= pts_data[:, 2]
        pts_data[:, 1] /= pts_data[:, 2]

        pts.data = pts_data[:, :-1]
        pts.W = W_new
        pts.H = H_new

        return pts


class RandomFlip(BaseTransform):
    """
    Performs a random flip of an image.

    """
    def __init__(self, p=0.5, axis=1):
        super(RandomFlip, self).__init__(p=p)
        self.params = None
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
    def __init__(self, rotation_range, padding='z', interpolation='bilinear', p=0.5):
        """
        Constructor.

        Parameters
        ----------
        rotation_range : rotation range
        p : probability of using this transform
        """
        super(RandomRotate, self).__init__(interpolation=interpolation, padding=padding, p=p)

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


class RandomShear(MatrixTransform):
    """
    Random shear around the center.

    """
    def __init__(self, range_x, range_y, interpolation='bilinear', padding='z', p=0.5):
        """
        Constructor.

        Parameters
        ----------
        range_x : shearing range along X-axis
        range_y : shearing range along Y-axis
        p : probability of using the transform
        """
        super(RandomShear, self).__init__(p=p, padding=padding, interpolation=interpolation)
        if range_x is None:
            range_x = 1
        if range_y is None:
            range_y = 1

        if str(range_x).isdigit():
            range_x = (range_x, range_x)

        if str(range_y).isdigit():
            range_y = (range_x, range_y)

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


class RandomScale(MatrixTransform):
    """
    Random scale transform.

    """
    def __init__(self, range_x=None, range_y=None, interpolation='bilinear', p=0.5):
        super(RandomScale, self).__init__(p=p, interpolation=interpolation)
        if range_x is None:
            range_x = 1
        if range_y is None:
            range_y = 1

        if str(range_x).isdigit():
            range_x = (range_x, range_x)

        if str(range_y).isdigit():
            range_y = (range_x, range_y)

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


class RandomCrop(BaseTransform):
    def __init__(self, crop_size, pad=None):
        super(RandomCrop, self).__init__(p=1)
        self.crop_size = crop_size

    def sample_transform(self):
        raise NotImplementedError

    @img_shape_checker
    def _apply_img(self, img):
        raise NotImplementedError

    def _apply_mask(self, mask):
        raise NotImplementedError

    def _apply_labels(self, labels):
        return labels

    def _apply_pts(self, pts):
        raise NotImplementedError


class RandomPerspective(MatrixTransform):
    def __init__(self, tilt_range, p=0.5):
        super(RandomPerspective, self).__init__(p)
        self.__tilt_range = tilt_range

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

