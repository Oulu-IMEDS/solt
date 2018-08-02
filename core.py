img_types = {'I', 'M', 'P','L'}

class Pipeline(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        # We can combine some of the transforms using stack, e.g Matrix transforms
        # In the future version of the pipeline

        for trf in self.trf:
            data = trf(data)

class DataContainer(object):
    def __init__(self, data, fmt):
        self.__data = data
        self.__fmt = fmt

    def __getitem__(self, idx):
        return self.__data[idx], self.__fmt[idx]

    def __len__(self):
        return len(self.data)

class BasicTransform(object):
    def __init__(self, interpolation='bilinear', p=0.5):
        self.p = p
        self.interp = interpolation

    def sample_transform(self):
        pass

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

    def __apply_img(self, data, params):
        return None

    def __apply_pts(self, data, params):
        return None

    def apply_mask(self, data, params):
        return None

    def apply_labels(self, data, params):
        return None


def MatrixTransform(BasicTransform):
    def __init__(self, params):
        raise NotImplementedError
