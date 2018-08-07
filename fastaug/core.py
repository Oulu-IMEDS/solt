from abc import ABCMeta, abstractmethod
from .data import DataContainer, img_shape_checker

import numpy as np
import cv2

class Pipeline(object):
    """
    Pipeline class. Executes the list of transformations

    """
    def __init__(self, transforms=None):
        """
        Class constructor.

        Parameters
        ----------
        transforms : list or None
            List of transforms to execute
        """
        if transforms is None:
            transforms = []
        self.transforms = transforms

    def __call__(self, data):
        """
        Executes the list of the pre-defined transformations for a given data container.

        Parameters
        ----------
        data : DataContainer
            Data to be augmented

        Returns
        -------
        out : DataContainer
            Result

        """
        # TODO: We can combine some of the transforms using stack, e.g Matrix transforms by pre-computig them
        # Each transform has sample_transform method
        return Pipeline.exec_pipeline(self.transforms, data)

    @staticmethod
    def exec_pipeline(transforms, data):
        """
        Static method, executes the list of transformations for a given data point.

        Parameters
        ----------
        transforms : list
            List of transformations to execute
        data : DataContainer
            Data to be augmented

        Returns
        -------
        out : DataContainer
            Result
        """
        for trf in transforms:
            assert isinstance(trf, Pipeline) or isinstance(trf, BaseTransform)
            if isinstance(trf, BaseTransform):
                if trf.use_transform():
                    trf.sample_transform()
                    data = trf.apply(data)
            else:
                data = trf(data)  # It means that the transform is actually a nested pipeline

        return data


class SelectivePipeline(Pipeline):
    """
    Pipeline, which uniformly selects n out of k given transforms.

    """
    def __init__(self, transforms=None, n=1):
        """
        Constructor.

        Parameters
        ----------
        transforms : list
            List of k transforms to sample from
        n : int
            How many transform to sample
        """
        super(SelectivePipeline, self).__init__(transforms)
        assert 0 < n <= len(self.transforms)
        self.n = n

    def __call__(self, data):
        """
        Applies randomly selected n transforms to the given data item

        Parameters
        ----------
        data : DataContainer
            Data to be augmented

        Returns
        -------
        out : DataContainer
            Result
        """
        if len(self.transforms) > 0:
            trfs = np.random.choice(self.transforms, self.n, replace=False, p=1./self.n)
            return Pipeline.exec_pipeline(trfs, data)
        return data


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
        mask : mdarray
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
    def __init__(self, interpolation='bilinear', p=0.5):
        super(MatrixTransform, self).__init__(p)
        self.interpolation = interpolation


    @img_shape_checker
    def _apply_img(self, img):
        """
        Applies a matrix transform to an image.

        """
        H, W, _ = img.shape
        origin  = [H // 2, W // 2]

        T_origin = np.array([1, 0, -origin[0],
                             0, 1, -origin[1],
                             0, 0, 1]).reshape((3, 3))

        T_origin_back = np.array([1, 0, origin[0],
                                  0, 1, origin[1],
                                  0, 0, 1]).reshape((3, 3))

        M = self.params['transform_matrix']

        return cv2.warpPerspective(img, T_origin_back @ M @ T_origin, (W, H))

    @staticmethod
    def change_transform_origin(M, origin):
        """
        Method takes a matrix transform, and modifies its origin.

        Parameters
        ----------
        M : ndarray
            Transform (3x3) matrix
        origin : tuple or list
            Origin, around which the transform needs to be applied

        Returns
        -------
        out : ndarray
            Modified Transform matrix

        """
        T_origin = np.array([1, 0, -origin[0],
                             0, 1, -origin[1],
                             0, 0, 1]).reshape((3, 3))

        T_origin_back = np.array([1, 0, origin[0],
                                  0, 1, origin[1],
                                  0, 0, 1]).reshape((3, 3))
        return T_origin_back @ M @ T_origin

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
        H, W, _ = mask.shape
        # X, Y coordinates
        M = self.params['transform_matrix']
        origin = (W // 2, H // 2)
        return cv2.warpPerspective(mask, MatrixTransform.change_transform_origin(M, origin), (W, H))

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
        pts_data = pts.data
        origin = [pts.W // 2, pts.H // 2]
        M = self.params['transform_matrix']
        M = MatrixTransform.change_transform_origin(M, origin)

        pts_data = np.hstack((pts_data, np.ones((pts_data.shape[0], 1))))
        pts_data = np.dot(M, pts_data.T).T

        pts_data[:, 0] /= pts_data[:, 2]
        pts_data[:, 1] /= pts_data[:, 2]

        pts.data = pts_data

        return pts