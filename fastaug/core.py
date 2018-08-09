from .data import DataContainer
from .transforms import BaseTransform

import numpy as np


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

