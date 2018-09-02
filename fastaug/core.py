import numpy as np

from collections import OrderedDict
from .data import DataContainer
from .base_transforms import BaseTransform, MatrixTransform


class Pipeline(object):
    """
    Pipeline class. Executes the list of transformations

    """
    def __init__(self, transforms=None, interpolation=None):
        # TODO: pipeline-wide interpolation and padding methods
        """
        Class constructor.

        Parameters
        ----------
        transforms : list or None
            List of transforms to execute
        """
        if transforms is None:
            transforms = []
        self.__transforms = transforms
        self.__interpolation = interpolation

    def serialize(self):
        """
        Serializes a pipeline into an OrderedDict

        Returns
        -------
        out : OrderedDict

        """
        res = OrderedDict()
        for t in self.__transforms:
            res[t.__class__.__name__] = t.serialize()

        return res

    @property
    def transforms(self):
        return self.__transforms

    @transforms.setter
    def transforms(self, value):
        self.__transforms = value

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
        return Pipeline.exec_pipeline(self.__transforms, data)

    @staticmethod
    def optimize_stack(transforms):
        """
        Static method which fuses the transformations

        Parameters
        ----------
        transforms : list
            A list of transforms

        Returns
        -------
        out : list
            An optimized list of transforms

        """
        # First we should create a stack
        transforms_stack = []
        for trf in transforms:
            assert isinstance(trf, Pipeline) or isinstance(trf, BaseTransform)
            if isinstance(trf, BaseTransform):
                if trf.use_transform():
                    trf.sample_transform()
                    if isinstance(trf, MatrixTransform):
                        if len(transforms_stack) == 0:
                            transforms_stack.append(trf)
                        else:
                            if isinstance(transforms_stack[-1], MatrixTransform):
                                transforms_stack[-1].fuse_with(trf)
                            else:
                                transforms_stack.append(trf)
                    else:
                        transforms_stack.append(trf)
            else:
                transforms_stack.append(trf)  # It means that the transform is actually a nested pipeline

        return transforms_stack

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

        # Performing the transforms using the optimized stack
        transforms = Pipeline.optimize_stack(transforms)
        for trf in transforms:
            if isinstance(trf, BaseTransform):
                data = trf.apply(data)
            elif isinstance(trf, Pipeline):
                data = trf(data)
            else:
                raise NotImplementedError
        return data


class SelectivePipeline(Pipeline):
    """
    Pipeline, which uniformly selects n out of k given transforms.

    """
    def __init__(self, transforms=None, n=1, probs=None):
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
        if probs is not None:
            assert len(probs) == n
        self.n = n
        self.probs = probs

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
            trfs = np.random.choice(self.transforms, self.n, replace=False, p=self.probs)
            trfs = Pipeline.optimize_stack(trfs)
            data = Pipeline.exec_pipeline(trfs, data)
        return data

