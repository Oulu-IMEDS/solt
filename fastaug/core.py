from abc import ABCMeta, abstractmethod

import numpy as np

img_types = {'I', 'M', 'P','L'}

class DataContainer(object):
    def __init__(self, data:tuple, fmt:str):
        if len(fmt) == 1 and not isinstance(data, tuple):
            data = (data,)
        self.__data = data
        self.__fmt = fmt

    def __getitem__(self, idx:int):
        return self.__data[idx], self.__fmt[idx]

    def __len__(self):
        return len(self.data)


class Pipeline(object):
    def __init__(self, transforms=None):
        if transforms is None:
            transforms = []
        self.transforms = transforms

    def __call__(self, data:DataContainer):
        # We can combine some of the transforms using stack, e.g Matrix transforms
        # In the future version of the pipeline
        return Pipeline.exec_pipeline(self.transforms, data)

    @staticmethod
    def exec_pipeline(transforms, data):
        for trf in transforms:
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


class BasicTransform(metaclass=ABCMeta):
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
        for (item, t) in data:
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

        return DataContainer(data=result, fmt=''.join(types))

    def __call__(self, data):
        if self.use_transform():
            self.sample_transform()
            return self.apply(data)
        else:
            return data

    @abstractmethod
    def _apply_img(self, data):
        pass

    @abstractmethod
    def _apply_pts(self, data):
        pass

    @abstractmethod
    def _apply_mask(self, data):
        pass

    @abstractmethod
    def _apply_labels(self, data):
        pass


class MatrixTransform(BasicTransform):
    def __init__(self, interpolation='bilinear', p=0.5):
        super(MatrixTransform, self).__init__(p)
        self.interpolation = interpolation