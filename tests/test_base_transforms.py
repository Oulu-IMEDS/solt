import solt.core as slc
import solt.data as sld
import solt.transforms as slt
import solt.base_transforms as slb
import numpy as np
import cv2
import pytest

from .fixtures import img_2x2, img_3x3, img_3x4, mask_2x2, mask_3x4, mask_3x3, img_5x5, img_6x6


def test_parameter_validation_raises_error_when_types_dont_match():
    with pytest.raises(NotImplementedError):
        slb.validate_parameter([1, 2], {1, 2}, 10, int)


def test_parameter_validation_raises_error_when_default_type_is_wrong():
    with pytest.raises(AssertionError):
        slb.validate_parameter(None, {1, 2}, (10, '12345'), int)


def test_parameter_validation_raises_error_when_default_value_is_wrong_type():
    with pytest.raises(AssertionError):
        slb.validate_parameter(None, {1, 2}, ('10', 'inherit'), int)


def test_data_indices_cant_be_list():
    with pytest.raises(AssertionError):
        slt.ImageSaltAndPepper(data_indices=[])


def test_data_indices_can_be_only_int():
    with pytest.raises(AssertionError):
        slt.ImageSaltAndPepper(data_indices=('2', 34))


def test_data_indices_can_be_only_nonnegative():
    with pytest.raises(AssertionError):
        slt.ImageSaltAndPepper(data_indices=(0, 1, -2))


def test_transform_returns_original_data_if_use_transform_is_false(img_2x2):
    dc = sld.DataContainer((img_2x2, ), 'I')
    trf = slt.ImageSaltAndPepper(p=0)
    res = trf(dc)
    np.testing.assert_array_equal(res.data[0], img_2x2)


def test_transform_returns_original_data_if_not_in_specified_indices(img_2x2, img_3x3, img_3x4, img_5x5):
    kpts_data = np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2))
    kpts = sld.KeyPoints(kpts_data, 3, 3)
    dc = sld.DataContainer((img_2x2, img_3x3, img_3x4, img_5x5, 1, kpts), 'IIIILP')
    trf = slt.RandomFlip(p=1, data_indices=(0, 1, 4))
    res = trf(dc)

    assert np.linalg.norm(res.data[0]-img_2x2) > 0
    assert np.linalg.norm(res.data[1]-img_3x3) > 0
    np.testing.assert_array_equal(res.data[2], img_3x4)
    np.testing.assert_array_equal(res.data[3], img_5x5)
    assert res.data[4] == 1
    np.testing.assert_array_equal(res.data[5].data, kpts_data)
