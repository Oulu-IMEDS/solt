import numpy as np
img_types = {'I', 'M', 'P','L'}

class DataContainer(object):
    def __init__(self, data, fmt):
        self.__data = data
        self.__fmt = fmt

    def __getitem__(self, idx):
        return self.__data[idx], self.__fmt[idx]

    def __len__(self):
        return len(self.data)


class Pipeline(object):
    def __init__(self, transforms=None):
        if transforms is None:
            transforms = []
        self.transforms = transforms

    def __call__(self, data):
        # We can combine some of the transforms using stack, e.g Matrix transforms
        # In the future version of the pipeline

        for trf in self.transforms:
            data = trf(data)

        return data


class BasicTransform(object):
    def __init__(self, p=0.5):
        self.p = p

    def use_transform(self):
        if np.random.rand() > self.p:
            return True
        return False

    def sample_transform(self):
        raise NotImplementedError

    def apply(self, data, params):
        result = []
        types = []
        for (item, t) in data:
            if t == 'I':
                tmp_item = self.__apply_img(item, params)
            if t == 'M':
                tmp_item = self.__apply_mask(item, params)
            if t == 'P':
                tmp_item = self.__apply_pts(item, params)
            if t == 'L':
                tmp_item = self.__apply_labels(item, params)
            types.append(t)
            result.append(tmp_item)

        return DataContainer(data=result, types=types)

    def __call__(self, data):
        if self.use_transform():
            params = self.sample_transform()
            return self.apply(data, params)
        else:
            data

    def __apply_img(self, data, params):
        raise NotImplementedError

    def __apply_pts(self, data, params):
        raise NotImplementedError

    def __apply_mask(self, data, params):
        raise NotImplementedError

    def __apply_labels(self, data, params):
        raise NotImplementedError


def MatrixTransform(BasicTransform):
    def __init__(self, interpolation='bilinear', p=0.5):
        super(MatrixTransform, self).__init__(p)
        self.interpolation = interpolation


class Selector(object):
    def __init__(self, transforms, n):
        self.transforms = transforms
        self.n = n

    def __call__(self, data):
        pass