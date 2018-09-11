import solt.core as slc
import solt.data as sld
import solt.transforms as slt
import solt.base_transforms as slb
import numpy as np
import cv2
import pytest

from .fixtures import img_2x2, img_3x3, img_3x4, mask_2x2, mask_3x4, mask_3x3, img_5x5, img_6x6


def test_img_mask_vertical_flip(img_3x4, mask_3x4):
    img, mask = img_3x4, mask_3x4
    dc = sld.DataContainer((img, mask), 'IM')

    stream = slc.Stream([
        slt.RandomFlip(p=1, axis=0)
    ])

    dc = stream(dc)
    img_res, _ = dc[0]
    mask_res, _ = dc[1]

    h, w = mask.shape
    assert np.array_equal(cv2.flip(img, 0).reshape(h, w, 1), img_res)
    assert np.array_equal(cv2.flip(mask, 0), mask_res)


def test_img_mask_mask_vertical_flip(img_3x4, mask_3x4):
    img, mask = img_3x4, mask_3x4
    dc = sld.DataContainer((img, mask, mask), 'IMM')

    stream = slc.Stream([
        slt.RandomFlip(p=1, axis=0)
    ])

    dc = stream(dc)
    img_res, _ = dc[0]
    mask_res, _ = dc[1]

    h, w = mask.shape
    assert np.array_equal(cv2.flip(img, 0).reshape(h, w, 1), img_res)
    assert np.array_equal(cv2.flip(mask, 0), mask_res)


def test_img_mask_horizontal_flip(img_3x4, mask_3x4):
    img, mask = img_3x4, mask_3x4
    dc = sld.DataContainer((img, mask), 'IM')

    stream = slc.Stream([
        slt.RandomFlip(p=1, axis=1)
    ])

    dc = stream(dc)
    img_res, _ = dc[0]
    mask_res, _ = dc[1]

    h, w = mask.shape
    assert np.array_equal(cv2.flip(img, 1).reshape(h, w, 1), img_res)
    assert np.array_equal(cv2.flip(mask, 1), mask_res)


def test_img_mask_vertical_horizontal_flip(img_3x4, mask_3x4):
    img, mask = img_3x4, mask_3x4
    dc = sld.DataContainer((img, mask), 'IM')

    stream = slc.Stream([
        slt.RandomFlip(p=1, axis=0),
        slt.RandomFlip(p=1, axis=1)
    ])

    dc = stream(dc)
    img_res, _ = dc[0]
    mask_res, _ = dc[1]

    h, w = mask.shape
    assert np.array_equal(cv2.flip(cv2.flip(img, 0), 1).reshape(h, w, 1), img_res)
    assert np.array_equal(cv2.flip(cv2.flip(mask, 0), 1), mask_res)


def test_keypoints_vertical_flip():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = sld.KeyPoints(kpts_data, 2, 2)
    stream = slt.RandomFlip(p=1, axis=0)
    dc = sld.DataContainer((kpts,), 'P')

    dc_res = stream(dc)

    assert np.array_equal(dc_res[0][0].data, np.array([[0, 1], [0, 0], [1, 1], [1, 0]]).reshape((4, 2)))


def test_keypoints_horizontal_flip_within_stream():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = sld.KeyPoints(kpts_data, 2, 2)
    stream = slc.Stream([
        slt.RandomFlip(p=1, axis=1)
        ])
    dc = sld.DataContainer((kpts,), 'P')

    dc_res = stream(dc)

    assert np.array_equal(dc_res[0][0].data, np.array([[1, 0], [1, 1], [0, 0], [0, 1]]).reshape((4, 2)))


def test_keypoints_vertical_flip_within_stream():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = sld.KeyPoints(kpts_data, 2, 2)
    stream = slc.Stream([
        slt.RandomFlip(p=1, axis=0)
        ])
    dc = sld.DataContainer((kpts,), 'P')

    dc_res = stream(dc)

    assert np.array_equal(dc_res[0][0].data, np.array([[0, 1], [0, 0], [1, 1], [1, 0]]).reshape((4, 2)))


def test_rotate_range_none():
    trf = slt.RandomRotate(None)
    assert trf.rotation_range == (0, 0)


def test_shear_range_none():
    trf = slt.RandomShear(None, None)
    assert trf.shear_range_x == (0, 0)
    assert trf.shear_range_y == (0, 0)


def test_rotate_90_img_mask_keypoints(img_3x3, mask_3x3):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2))
    kpts = sld.KeyPoints(kpts_data, 3, 3)
    img, mask = img_3x3, mask_3x3
    H, W = mask.shape

    dc = sld.DataContainer((img, mask, kpts, 1), 'IMPL')
    # Defining the 90 degrees transform (clockwise)
    stream = slt.RandomRotate(rotation_range=(90, 90), p=1)
    dc_res = stream(dc)

    img_res, _ = dc_res[0]
    mask_res, _ = dc_res[1]
    kpts_res, _ = dc_res[2]
    label_res, _ = dc_res[3]
    M = cv2.getRotationMatrix2D((W // 2, H // 2), -90, 1)
    expected_img_res = cv2.warpAffine(img, M, (W, H)).reshape((H, W, 1))
    expected_mask_res = cv2.warpAffine(mask, M, (W, H))
    expected_kpts_res = np.array([[2, 0], [0, 0], [0, 2], [2, 2]]).reshape((4, 2))

    assert np.array_equal(expected_img_res, img_res)
    assert np.array_equal(expected_mask_res, mask_res)
    np.testing.assert_array_almost_equal(expected_kpts_res, kpts_res.data)
    assert label_res == 1

def test_zoom_x_axis_odd(img_5x5):
    stream = slt.RandomScale(range_x=(0.5, 0.5), range_y=(1, 1), same=False, p=1)
    dc = sld.DataContainer((img_5x5,), 'I')
    H, W = img_5x5.shape[0], img_5x5.shape[1]
    img_res = stream(dc)[0][0]
    assert H == img_res.shape[0]
    assert W // 2 == img_res.shape[1]


def test_scale_x_axis_even(img_6x6):
    stream = slt.RandomScale((0.5, 0.5), (1, 1), same=False, p=1)
    dc = sld.DataContainer((img_6x6,), 'I')
    H, W = img_6x6.shape[0], img_6x6.shape[1]
    img_res = stream(dc)[0][0]
    assert H == img_res.shape[0]
    assert W // 2 == img_res.shape[1]


def test_scale_xy_axis_odd(img_5x5):
    stream = slt.RandomScale((0.5, 0.5), (3, 3), same=False, p=1)
    dc = sld.DataContainer((img_5x5,), 'I')
    H, W = img_5x5.shape[0], img_5x5.shape[1]
    img_res = stream(dc)[0][0]
    assert H * 3 == img_res.shape[0]
    assert W // 2 == img_res.shape[1]


def test_scale_xy_axis_even(img_6x6):
    stream = slt.RandomScale((0.5, 0.5), (2, 2), same=False, p=1)
    dc = sld.DataContainer((img_6x6,), 'I')
    H, W = img_6x6.shape[0], img_6x6.shape[1]
    img_res = stream(dc)[0][0]
    assert H * 2 == img_res.shape[0]
    assert W // 2 == img_res.shape[1]


def test_scale_img_mask(img_3x4, mask_3x4):
    img_mask_3x4 = img_3x4, mask_3x4
    stream = slt.RandomScale((0.5, 0.5), (2, 2), same=False, p=1)
    dc = sld.DataContainer(img_mask_3x4, 'IM')
    H, W = img_mask_3x4[0].shape[0], img_mask_3x4[0].shape[1]
    img_res = stream(dc)[0][0]
    mask_res = stream(dc)[1][0]
    assert H * 2 == img_res.shape[0],  W // 2 == img_res.shape[1]
    assert H * 2 == mask_res.shape[0],  W // 2 == mask_res.shape[1]


def test_keypoints_assert_reflective(img_3x3, mask_3x3):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2))
    kpts = sld.KeyPoints(kpts_data, 3, 3)
    img, mask = img_3x3, mask_3x3

    dc = sld.DataContainer((img, mask, kpts,), 'IMP')
    # Defining the 90 degrees transform (clockwise)
    stream = slt.RandomRotate(rotation_range=(20, 20), p=1, padding='r')
    with pytest.raises(ValueError):
        stream(dc)


def test_padding_img_2x2_4x4(img_2x2):
    img = img_2x2
    dc = sld.DataContainer((img,), 'I')
    transf = slt.PadTransform((4, 4))
    res = transf(dc)
    assert (res[0][0].shape[0] == 4) and (res[0][0].shape[1] == 4)


def test_padding_img_2x2_2x2(img_2x2):
    img = img_2x2
    dc = sld.DataContainer((img,), 'I')
    transf = slt.PadTransform((2, 2))
    res = transf(dc)
    assert (res[0][0].shape[0] == 2) and (res[0][0].shape[1] == 2)


def test_padding_img_mask_2x2_4x4(img_2x2, mask_2x2):
    img, mask = img_2x2, mask_2x2
    dc = sld.DataContainer((img, mask), 'IM')
    transf = slt.PadTransform((4, 4))
    res = transf(dc)
    assert (res[0][0].shape[0] == 4) and (res[0][0].shape[1] == 4)
    assert (res[1][0].shape[0] == 4) and (res[1][0].shape[1] == 4)


def test_padding_img_2x2_3x3(img_2x2):
    img = img_2x2
    dc = sld.DataContainer((img,), 'I')
    transf = slt.PadTransform((3, 3))
    res = transf(dc)
    assert (res[0][0].shape[0] == 3) and (res[0][0].shape[1] == 3)


def test_padding_img_mask_2x2_3x3(img_2x2, mask_2x2):
    img, mask = img_2x2, mask_2x2
    dc = sld.DataContainer((img, mask), 'IM')
    transf = slt.PadTransform((3, 3))
    res = transf(dc)
    assert (res[0][0].shape[0] == 3) and (res[0][0].shape[1] == 3)
    assert (res[1][0].shape[0] == 3) and (res[1][0].shape[1] == 3)


def test_padding_img_mask_3x4_3x4(img_3x4, mask_3x4):
    img, mask = img_3x4, mask_3x4
    dc = sld.DataContainer((img, mask), 'IM')
    transf = slt.PadTransform((4, 3))
    res = transf(dc)
    assert (res[0][0].shape[0] == 3) and (res[0][0].shape[1] == 4)
    assert (res[1][0].shape[0] == 3) and (res[1][0].shape[1] == 4)


def test_padding_img_mask_3x4_5x5(img_3x4, mask_3x4):
    img, mask = img_3x4, mask_3x4
    dc = sld.DataContainer((img, mask), 'IM')
    transf = slt.PadTransform((5, 5))
    res = transf(dc)
    assert (res[0][0].shape[0] == 5) and (res[0][0].shape[1] == 5)
    assert (res[1][0].shape[0] == 5) and (res[1][0].shape[1] == 5)


def test_pad_to_20x20_img_mask_keypoints_3x3(img_3x3, mask_3x3):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2))
    kpts = sld.KeyPoints(kpts_data, 3, 3)
    img, mask = img_3x3, mask_3x3

    dc = sld.DataContainer((img, mask, kpts,), 'IMP')
    transf = slt.PadTransform((20, 20))
    res = transf(dc)

    assert (res[0][0].shape[0] == 20) and (res[0][0].shape[1] == 20)
    assert (res[1][0].shape[0] == 20) and (res[1][0].shape[1] == 20)
    assert (res[2][0].H == 20) and (res[2][0].W == 20)

    assert np.array_equal(res[2][0].data, np.array([[8, 8], [8, 10], [10, 10], [10, 8]]).reshape((4, 2)))


def test_pad_to_20x20_img_mask_keypoints_3x3_kpts_first(img_3x3, mask_3x3):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2))
    kpts = sld.KeyPoints(kpts_data, 3, 3)
    img, mask = img_3x3, mask_3x3

    dc = sld.DataContainer((kpts, img, mask), 'PIM')
    transf = slt.PadTransform((20, 20))
    res = transf(dc)

    assert (res[2][0].shape[0] == 20) and (res[2][0].shape[1] == 20)
    assert (res[1][0].shape[0] == 20) and (res[1][0].shape[1] == 20)
    assert (res[0][0].H == 20) and (res[0][0].W == 20)

    assert np.array_equal(res[0][0].data, np.array([[8, 8], [8, 10], [10, 10], [10, 8]]).reshape((4, 2)))


def test_3x3_pad_to_20x20_center_crop_3x3_shape_stayes_unchanged(img_3x3, mask_3x3):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2))
    kpts = sld.KeyPoints(kpts_data, 3, 3)
    img, mask = img_3x3, mask_3x3

    dc = sld.DataContainer((img, mask, kpts,), 'IMP')

    stream = slc.Stream([
        slt.PadTransform((20, 20)),
        slt.CropTransform((3, 3))
    ])
    res = stream(dc)

    assert (res[0][0].shape[0] == 3) and (res[0][0].shape[1] == 3)
    assert (res[1][0].shape[0] == 3) and (res[1][0].shape[1] == 3)
    assert (res[2][0].H == 3) and (res[2][0].W == 3)


@pytest.mark.parametrize('pad_size,crop_size', [
    (20, 2),
    (20, (2, 2)),
    ((20,20), (2, 2)),
    ((20, 20), 2),
    ]
)
def test_2x2_pad_to_20x20_center_crop_2x2(pad_size, crop_size, img_2x2, mask_2x2):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 1], [1, 1], [1, 0]]).reshape((4, 2))
    kpts = sld.KeyPoints(kpts_data, 2, 2)
    img, mask = img_2x2, mask_2x2

    dc = sld.DataContainer((img, mask, kpts,), 'IMP')

    stream = slc.Stream([
        slt.PadTransform(pad_to=pad_size),
        slt.CropTransform(crop_size=crop_size)
    ])
    res = stream(dc)

    assert (res[0][0].shape[0] == 2) and (res[0][0].shape[1] == 2)
    assert (res[1][0].shape[0] == 2) and (res[1][0].shape[1] == 2)
    assert (res[2][0].H == 2) and (res[2][0].W == 2)

    assert np.array_equal(res[0][0], img)
    assert np.array_equal(res[1][0], mask)
    assert np.array_equal(res[2][0].data, kpts_data)


@pytest.mark.parametrize('crop_mode', [
    'c',
    'r',
    'd'
    ]
)
def test_different_crop_modes(crop_mode, img_2x2, mask_2x2):
    if crop_mode == 'd':
        with pytest.raises(ValueError):
            slt.CropTransform(crop_size=2, crop_mode=crop_mode)
    else:
        stream = slc.Stream([
            slt.PadTransform(pad_to=20),
            slt.CropTransform(crop_size=2, crop_mode=crop_mode)
        ])
        img, mask = img_2x2, mask_2x2
        dc = sld.DataContainer((img, mask,), 'IM')
        dc_res = stream(dc)

        for el in dc_res.data:
            assert el.shape[0] == 2
            assert el.shape[1] == 2


def test_6x6_pad_to_20x20_center_crop_6x6_img_kpts(img_6x6):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 5], [1, 3], [2, 0]]).reshape((4, 2))
    kpts = sld.KeyPoints(kpts_data, 6, 6)
    img = img_6x6

    dc = sld.DataContainer((img, kpts,1), 'IPL')

    stream = slc.Stream([
        slt.PadTransform((20, 20)),
        slt.CropTransform((6, 6))
    ])
    res = stream(dc)

    assert (res[0][0].shape[0] == 6) and (res[0][0].shape[1] == 6)
    assert (res[1][0].H == 6) and (res[1][0].W == 6)

    assert np.array_equal(res[0][0], img)
    assert np.array_equal(res[1][0].data, kpts_data)


def test_6x6_pad_to_20x20_center_crop_6x6_kpts_img(img_6x6):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 5], [1, 3], [2, 0]]).reshape((4, 2))
    kpts = sld.KeyPoints(kpts_data, 6, 6)
    img = img_6x6

    dc = sld.DataContainer((kpts, img), 'PI')

    stream = slc.Stream([
        slt.PadTransform((20, 20)),
        slt.CropTransform((6, 6))
    ])
    res = stream(dc)

    assert (res[1][0].shape[0] == 6) and (res[1][0].shape[1] == 6)
    assert (res[0][0].H == 6) and (res[0][0].W == 6)

    assert np.array_equal(res[1][0], img)
    assert np.array_equal(res[0][0].data, kpts_data)


def test_translate_forward_backward_sampling():
    stream = slc.Stream([
        slt.RandomTranslate(range_x=(1, 1), range_y=(1, 1), p=1),
        slt.RandomTranslate(range_x=(-1, -1), range_y=(-1, -1), p=1),
    ])
    trf = stream.optimize_stack(stream.transforms)[0]
    assert 1 == trf.state_dict['translate_x']  # The settings will be overrided by the first transform
    assert 1 == trf.state_dict['translate_y']  # The settings will be overrided by the first transform
    assert np.array_equal(trf.state_dict['transform_matrix'], np.eye(3))


def test_projection_empty_sampling():
    trf = slt.RandomProjection(p=1)
    trf.sample_transform()
    assert np.array_equal(trf.state_dict['transform_matrix'], np.eye(3))


def test_projection_translate_plus_minus_1():
    trf = slt.RandomProjection(affine_transforms=slc.Stream([
        slt.RandomTranslate(range_x=(1, 1), range_y=(1, 1), p=1),
        slt.RandomTranslate(range_x=(-1, -1), range_y=(-1, -1), p=1),
    ]), p=1)

    trf.sample_transform()
    assert np.array_equal(trf.state_dict['transform_matrix'], np.eye(3))


def test_gaussian_noise_no_image_throws_value_error():
    trf = slt.ImageAdditiveGaussianNoise(p=1)
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 5], [1, 3], [2, 0]]).reshape((4, 2))
    kpts = sld.KeyPoints(kpts_data, 6, 6)
    dc = sld.DataContainer((kpts, ), 'P')

    with pytest.raises(ValueError):
        trf(dc)


def test_gaussian_noise_float_gain():
    trf = slt.ImageAdditiveGaussianNoise(gain_range=0.2, p=1)
    assert isinstance(trf._gain_range, tuple)
    assert len(trf._gain_range) == 2
    assert trf._gain_range[0] == 0 and trf._gain_range[1] == 0.2


def test_salt_and_pepper_no_gain(img_6x6):
    trf = slt.ImageSaltAndPepper(gain_range=0., p=1)
    dc_res = trf(sld.DataContainer((img_6x6.astype(np.uint8),), 'I'))
    assert np.array_equal(img_6x6, dc_res[0][0])


@pytest.mark.parametrize('pad_size,pad_type', [
    (2, 'z'),
    ((2, 2), 'z'),
    (2, 'r'),
    ((2, 2), 'r'),
    ]
)
def test_pad_does_not_change_the_data_when_the_image_and_the_mask_are_big(pad_size, pad_type, img_3x3, mask_3x3):
    dc = sld.DataContainer((img_3x3, mask_3x3), 'IM')
    trf = slt.PadTransform(pad_to=pad_size, padding=pad_type)
    dc_res = trf(dc)

    np.testing.assert_array_equal(dc_res.data[0], img_3x3)
    np.testing.assert_array_equal(dc_res.data[1], mask_3x3)


def test_image_doesnt_change_when_gain_0_in_gaussian_noise_addition(img_3x3):
    dc = sld.DataContainer((img_3x3, ), 'I')
    trf = slt.ImageAdditiveGaussianNoise(gain_range=(0, 0), p=1)
    dc_res = trf(dc)
    np.testing.assert_array_equal(img_3x3, dc_res.data[0])


@pytest.mark.parametrize('scale,expected', [
    (2, (1, 2)),
    (2.5, (1, 2.5)),
    (0.5, (0.5, 1)),
    (-1, None)
    ]
)
def test_scale_range_from_number(scale, expected):
    if expected is not None:
        trf = slt.RandomScale(range_x=scale, range_y=scale)
        assert trf.scale_range_x == expected
        assert trf.scale_range_x == expected
    else:
        with pytest.raises(ValueError):
            slt.RandomScale(range_x=scale)
        with pytest.raises(ValueError):
            slt.RandomScale(range_x=None, range_y=scale)


@pytest.mark.parametrize('translate,expected', [
    (2, (-2, 2)),
    (2.5, (-2.5, 2.5)),
    (0.5, (-0.5, 0.5)),
    (-0.5, (-0.5, 0.5)),
    ]
)
def test_translate_range_from_number(translate, expected):
    trf = slt.RandomTranslate(range_x=translate, range_y=translate)
    assert trf.translate_range_x == expected
    assert trf.translate_range_y == expected


@pytest.mark.parametrize('trf_cls,trf_params', [
    (slt.ImageAdditiveGaussianNoise, {'gain_range': 0.5, }),
    (slt.ImageSaltAndPepper, {'p': 1}),
    (slt.ImageGammaCorrection, {'p':1})
    ]
)
def test_image_trfs_dont_change_mask_labels_kpts(trf_cls, trf_params, img_3x4, mask_3x4):
    trf = trf_cls(**trf_params)
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [2, 0]]).reshape((4, 2))
    kpts = sld.KeyPoints(kpts_data, 3, 4)
    dc = sld.DataContainer((img_3x4, mask_3x4, kpts, 1), 'IMPL')
    dc_res = trf(dc)

    assert np.all(dc.data[1] == dc_res.data[1])
    assert np.all(dc.data[2].data == dc_res.data[2].data)
    assert dc.data[3] == dc_res.data[3]


def test_padding_cant_be_float():
    with pytest.raises(TypeError):
        slt.PadTransform(pad_to=2.5)


def test_reflective_padding_cant_be_applied_to_kpts():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [2, 0]]).reshape((4, 2))
    kpts = sld.KeyPoints(kpts_data, 3, 4)
    dc = sld.DataContainer((1, kpts), 'LP')
    trf = slt.PadTransform(pad_to=(10, 10), padding='r')
    with pytest.raises(ValueError):
        trf(dc)


@pytest.mark.parametrize('crop_size', [
    (2, 3),
    (3, 2),
    ]
)
def test_crop_size_is_too_big(img_2x2, crop_size):
    dc = sld.DataContainer((img_2x2,), 'I')
    trf = slt.CropTransform(crop_size=crop_size)
    with pytest.raises(ValueError):
        trf(dc)


@pytest.mark.parametrize('crop_size', [
    '123',
    2.5,
    (2.5, 2),
    (2, 2.2)
]
)
def test_wrong_crop_size_types(crop_size):
    with pytest.raises(TypeError):
        slt.CropTransform(crop_size=crop_size)


@pytest.mark.parametrize('salt_p', [
    (1, 2),
    (2, 2),
    (0.7, 0.3),
    (-0.1),
    (-0.8, 0.1),
    (0.3, 0.7, 0.8),
]
)
def test_wrong_salt_p_salt_and_pepper(salt_p):
    with pytest.raises(ValueError):
        slt.ImageSaltAndPepper(salt_p=salt_p)


@pytest.mark.parametrize('gain_range', [
    (1, 2),
    (2, 2),
    (0.7, 0.3),
    (-0.1),
    (-0.8, 0.1),
    (0.3, 0.7, 0.8),
]
)
def test_wrong_gain_range_vals_salt_and_pepper(gain_range):
    with pytest.raises(ValueError):
        slt.ImageSaltAndPepper(gain_range=gain_range)


def test_wrong_types_in_gain_and_salt_p_salt_and_peper():
    with pytest.raises(TypeError):
        slt.ImageSaltAndPepper(salt_p='2')

    with pytest.raises(TypeError):
        slt.ImageSaltAndPepper(gain_range='2')


@pytest.mark.parametrize('scale_x, scale_y, same, expected', [
    (None, (2,2), True, (2, 2)),
    ((2,2), None, True, (2, 2)),
    (None, None, True, (1, 1)),
    (None, (2,2), False, (1, 2)),
    ((2,2), None, False, (2, 1)),
    (None, None, False, (1, 1)),
]
)
def test_scale_when_range_x_is_none(scale_x, scale_y, same, expected):
    trf = slt.RandomScale(range_x=scale_x, range_y=scale_y, same=same, p=1)
    trf.sample_transform()
    assert (trf.state_dict['scale_x'], trf.state_dict['scale_y']) == expected


@pytest.mark.parametrize('translate_x, translate_y, expected', [
    (None, (2, 2), (0, 2)),
    ((2, 2), None, (2, 0)),
    (None, None, (0, 0)),
]
)
def test_scale_when_range_x_is_none(translate_x, translate_y, expected):
    trf = slt.RandomTranslate(range_x=translate_x, range_y=translate_y, p=1)
    trf.sample_transform()
    assert (trf.state_dict['translate_x'], trf.state_dict['translate_y']) == expected
