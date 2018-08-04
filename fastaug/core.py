from abc import ABCMeta, abstractmethod
from functools import wraps

import numpy as np

img_types = {'I', 'M', 'P','L'}


def img_shape_checker(method):
    """

    Parameters
    ----------
    transform : _apply_img method of BaseTransform

    Returns
    -------

    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        res = method(self, *args, **kwargs)
        assert 1 < len(res.shape) <= 3
        if len(res.shape) == 2:
            h, w = res.shape
            c = 1
        else:
            h, w, c = res.shape

        return res.reshape((h, w, c))
    return wrapper


class DataContainer(object):
    def __init__(self, data:tuple, fmt:str):
        if len(fmt) == 1 and not isinstance(data, tuple):
            data = (data,)
        self.__data = data
        self.__fmt = fmt

    def __getitem__(self, idx:int):
        return self.__data[idx], self.__fmt[idx]

    def __len__(self):
        return len(self.__data)


class Pipeline(object):
    def __init__(self, transforms=None):
        if transforms is None:
            transforms = []
        self.transforms = transforms

    def __call__(self, data:DataContainer):
        # TODO: We can combine some of the transforms using stack, e.g Matrix transforms by precomputig them
        # Each transform has sample_transform method
        return Pipeline.exec_pipeline(self.transforms, data)

    @staticmethod
    def exec_pipeline(transforms, data):
        for trf in transforms:
            assert isinstance(trf, Pipeline) or isinstance(trf, BaseTransform)
            data = trf(data)

        return data


class SelectivePipeline(Pipeline):
    def __init__(self, transforms=None, n=1):
        super(SelectivePipeline, self).__init__(transforms)
        assert n > 0
        self.n = n

    def __call__(self, data):
        if len(self.transforms) > 0:
            trfs = np.random.choice(self.transforms, self.n, replace=False, p=1./self.n)
            return Pipeline.exec_pipeline(trfs, data)
        return data


class BaseTransform(metaclass=ABCMeta):
    def __init__(self, p=0.5):
        self.p = p

    def use_transform(self):
        if np.random.rand() < self.p:
            return True
        return False

    @abstractmethod
    def sample_transform(self):
        pass

    def apply(self, data):
        result = []
        types = []
        for i, (item, t) in enumerate(data):
            if t == 'I':
                tmp_item = self._apply_img(item)
            elif t == 'M':
                tmp_item = self._apply_mask(item)
            elif t == 'P':
                tmp_item = self._apply_pts(item)
            else: # t==L
                tmp_item = self._apply_labels(item)

            types.append(t)
            result.append(tmp_item)

        return DataContainer(data=tuple(result), fmt=''.join(types))

    def __call__(self, data):
        if self.use_transform():
            self.sample_transform()
            return self.apply(data)
        else:
            return data

    @abstractmethod
    def _apply_img(self, img):
        pass

    @abstractmethod
    def _apply_mask(self, mask):
        pass

    @abstractmethod
    def _apply_labels(self, labels):
        pass

    @abstractmethod
    def _apply_pts(self, pts):
        pass


class MatrixTransform(BaseTransform):
    def __init__(self, interpolation='bilinear', p=0.5):
        super(MatrixTransform, self).__init__(p)
        self.interpolation = interpolation

    @abstractmethod
    @img_shape_checker
    def _apply_img(self, pts):
        pass

    @abstractmethod
    def _apply_pts(self, pts):
        pass

    @abstractmethod
    def _apply_mask(self, mask):
        pass

    @abstractmethod
    def _apply_labels(self, labels):
        pass


