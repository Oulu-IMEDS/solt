import solt.data as sld
import solt.transforms as slt
import solt.base_transforms as slb
import numpy as np
import pytest

from .fixtures import img_2x2, img_3x3, img_3x4, img_6x6, img_5x5


def test_parameter_validation_raises_error_when_types_dont_match():
    with pytest.raises(NotImplementedError):
        slb.validate_parameter([1, 2], {1, 2}, 10, int)


def test_parameter_validation_raises_error_when_default_type_is_wrong():
    with pytest.raises(ValueError):
        slb.validate_parameter(None, {1, 2}, (10, '12345'), int)


def test_parameter_validation_raises_error_when_default_value_is_wrong_type():
    with pytest.raises(TypeError):
        slb.validate_parameter(None, {1, 2}, ('10', 'inherit'), int)


def test_data_indices_cant_be_list():
    with pytest.raises(TypeError):
        slt.ImageSaltAndPepper(data_indices=[])


def test_data_indices_can_be_only_int():
    with pytest.raises(TypeError):
        slt.ImageSaltAndPepper(data_indices=('2', 34))


def test_data_indices_can_be_only_nonnegative():
    with pytest.raises(ValueError):
        slt.ImageSaltAndPepper(data_indices=(0, 1, -2))


def test_transform_returns_original_data_if_use_transform_is_false(img_2x2):
    dc = sld.DataContainer((img_2x2, ), 'I')
    trf = slt.ImageSaltAndPepper(p=0)
    res = trf(dc)
    np.testing.assert_array_equal(res.data[0], img_2x2)


def test_transform_returns_original_data_if_not_in_specified_indices(img_2x2, img_3x3, img_3x4, img_5x5):
    kpts_data = np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2))
    kpts = sld.KeyPoints(kpts_data, 3, 3)
    dc = sld.DataContainer((img_2x2, img_3x3, img_3x4, img_5x5, 1, kpts, 2), 'IIIILPL')
    trf = slt.RandomFlip(p=1, data_indices=(0, 1, 4))
    res = trf(dc)

    assert np.linalg.norm(res.data[0]-img_2x2) > 0
    assert np.linalg.norm(res.data[1]-img_3x3) > 0
    np.testing.assert_array_equal(res.data[2], img_3x4)
    np.testing.assert_array_equal(res.data[3], img_5x5)
    assert res.data[-1] == 2
    np.testing.assert_array_equal(res.data[5].data, kpts_data)


@pytest.mark.parametrize('trf_cls,trf_params', [
    (slt.ImageAdditiveGaussianNoise, {'gain_range': 0.5, }),
    (slt.ImageSaltAndPepper, {'p': 1}),
    (slt.CropTransform, {'crop_size': 1}),
    (slt.PadTransform, {'pad_to': 1}),
    ]
)
def test_data_dependent_samplers_raise_nie_when_sample_transform_is_called(trf_cls, trf_params):
    with pytest.raises(NotImplementedError):
        trf = trf_cls(**trf_params)
        trf.sample_transform()


@pytest.mark.parametrize('img_1,img_2', [
    (img_2x2, img_6x6),
    (img_3x3, img_3x4),
    ]
)
def test_data_dep_trf_raises_value_error_when_imgs_are_of_different_size(img_1, img_2):
    trf = slt.ImageSaltAndPepper(gain_range=0., p=1)
    with pytest.raises(ValueError):
        trf(sld.DataContainer((1, img_1().astype(np.uint8), img_2().astype(np.uint8),), 'LII'))


@pytest.mark.parametrize('parameter', [
    (1, 2, 3),
    (10, 'inherit'),
    (1, 'i'),
    ]
)
def test_validate_parameter_raises_value_errors(parameter):
    with pytest.raises(ValueError):
        slb.validate_parameter(parameter, {1, 2}, 1, basic_type=int)


def test_transform_returns_original_data_when_not_used_and_applied(img_2x2):
    trf = slt.RandomFlip(p=0)
    dc = sld.DataContainer(img_2x2, 'I')
    dc_res = trf(dc)
    assert dc_res == dc


def test_transforms_are_serialized_with_state_when_needed():
    trf = slt.RandomRotate(rotation_range=(-90, 90))

    serialized = trf.serialize(include_state=True)

    assert 'dict' in serialized
    np.testing.assert_array_equal(serialized['dict']['transform_matrix'], np.eye(3))
