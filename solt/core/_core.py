import numpy as np

from collections import OrderedDict
from ..base_transforms import BaseTransform, MatrixTransform, DataDependentSamplingTransform
import copy
import random


class Stream(object):
    """
    Stream class. Executes the list of transformations

    """
    def __init__(self, transforms=None, interpolation=None, padding=None):
        """
        Class constructor.

        Parameters
        ----------
        transforms : list or None
            List of transforms to execute
        interpolation : str or None
            Stream-wide settings for interpolation. If for some particular transform your would like
            to still use its own mode, simply pass (<interpolation_value>, 'strict')
            in the constructor of that transform.
        padding : str or None
            Stream-wide settings for padding. If for some particular transform your would like
            to still use its own mode, simply pass (<padding_value>, 'strict')
            in the constructor of that transform.

        """
        if transforms is None:
            transforms = []

        for trf in transforms:
            if not isinstance(trf, BaseTransform) and not isinstance(trf, Stream):
                raise TypeError

        self.__interpolation = interpolation
        self.__padding = padding
        self.__transforms = transforms
        self._reset_stream_settings()

    @property
    def interpolation(self):
        return self.__interpolation

    @interpolation.setter
    def interpolation(self, value):
        self.__interpolation = value
        self._reset_stream_settings()

    @property
    def padding(self):
        return self.__padding

    @padding.setter
    def padding(self, value):
        self.__padding = value
        self._reset_stream_settings()

    def _reset_stream_settings(self):
        """
        Protected method, resets stream's settings

        """
        for trf in self.__transforms:
            if self.__interpolation is not None and hasattr(trf, 'interpolation'):
                if isinstance(trf, BaseTransform):
                    if trf.interpolation[1] != 'strict':
                        trf.interpolation = (self.__interpolation, trf.interpolation[1])
                elif isinstance(trf, Stream):
                    trf.interpolation = self.interpolation

            if self.__padding is not None and hasattr(trf, 'padding'):
                if isinstance(trf, BaseTransform):
                    if trf.padding[1] != 'strict':
                        trf.padding = (self.__padding, trf.padding[1])
                elif isinstance(trf, Stream):
                    trf.padding = self.__padding

    def serialize(self):
        """
        Serializes a Stream into an OrderedDict

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
        return Stream.exec_stream(self.__transforms, data)

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
            if not isinstance(trf, Stream) and not isinstance(trf, BaseTransform):
                raise TypeError
            if isinstance(trf, BaseTransform) and not isinstance(trf, DataDependentSamplingTransform):
                trf.reset_state()
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
                transforms_stack.append(trf)  # It means that the transform is actually a nested Stream

        return transforms_stack

    @staticmethod
    def exec_stream(transforms, data):
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
        transforms = Stream.optimize_stack(transforms)
        for trf in transforms:
            if isinstance(trf, BaseTransform) and not isinstance(trf, DataDependentSamplingTransform):
                data = trf.apply(data)
            elif isinstance(trf, Stream) or isinstance(trf, DataDependentSamplingTransform):
                data = trf(data)
        return data


class SelectiveStream(Stream):
    """
    Stream, which uniformly selects n out of k given transforms.

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
        super(SelectiveStream, self).__init__(transforms)
        if transforms is None:
            transforms = []
        if n < 0 or n > len(transforms):
            raise ValueError
        if probs is not None:
            if len(probs) != len(transforms):
                raise ValueError
        self._n = n
        self._probs = probs

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
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
            trfs = random_state.choice(self.transforms, self._n, replace=False, p=self._probs)
            trfs = [copy.deepcopy(x) for x in trfs]
            trfs = Stream.optimize_stack(trfs)
            data = Stream.exec_stream(trfs, data)
        return data

