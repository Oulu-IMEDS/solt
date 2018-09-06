import solt.core as augs_core
import solt.data as augs_data
import solt.transforms as trf
import numpy as np
import cv2
import pytest

from .fixtures import img_2x2, img_3x4, img_mask_2x2, img_mask_3x4, img_mask_3x3, img_5x5, img_6x6


def test_img_mask_vertical_flip(img_mask_3x4):
    img, mask = img_mask_3x4
    dc = augs_data.DataContainer((img, mask), 'IM')

    stream = augs_core.Stream([
        trf.RandomFlip(p=1, axis=0)
    ])

    dc = stream(dc)
    img_res, _ = dc[0]
    mask_res, _ = dc[1]

    h, w = mask.shape
    assert np.array_equal(cv2.flip(img, 0).reshape(h, w, 1), img_res)
    assert np.array_equal(cv2.flip(mask, 0), mask_res)


def test_img_mask_mask_vertical_flip(img_mask_3x4):
    img, mask = img_mask_3x4
    dc = augs_data.DataContainer((img, mask, mask), 'IMM')

    stream = augs_core.Stream([
        trf.RandomFlip(p=1, axis=0)
    ])

    dc = stream(dc)
    img_res, _ = dc[0]
    mask_res, _ = dc[1]

    h, w = mask.shape
    assert np.array_equal(cv2.flip(img, 0).reshape(h, w, 1), img_res)
    assert np.array_equal(cv2.flip(mask, 0), mask_res)


def test_img_mask_horizontal_flip(img_mask_3x4):
    img, mask = img_mask_3x4
    dc = augs_data.DataContainer((img, mask), 'IM')

    stream = augs_core.Stream([
        trf.RandomFlip(p=1, axis=1)
    ])

    dc = stream(dc)
    img_res, _ = dc[0]
    mask_res, _ = dc[1]

    h, w = mask.shape
    assert np.array_equal(cv2.flip(img, 1).reshape(h, w, 1), img_res)
    assert np.array_equal(cv2.flip(mask, 1), mask_res)


def test_img_mask_vertical_horizontal_flip(img_mask_3x4):
    img, mask = img_mask_3x4
    dc = augs_data.DataContainer((img, mask), 'IM')

    stream = augs_core.Stream([
        trf.RandomFlip(p=1, axis=0),
        trf.RandomFlip(p=1, axis=1)
    ])

    dc = stream(dc)
    img_res, _ = dc[0]
    mask_res, _ = dc[1]

    h, w = mask.shape
    assert np.array_equal(cv2.flip(cv2.flip(img, 0), 1).reshape(h, w, 1), img_res)
    assert np.array_equal(cv2.flip(cv2.flip(mask, 0), 1), mask_res)


def test_keypoints_vertical_flip():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = augs_data.KeyPoints(kpts_data, 2, 2)
    stream = trf.RandomFlip(p=1, axis=0)
    dc = augs_data.DataContainer((kpts, ), 'P')

    dc_res = stream(dc)

    assert np.array_equal(dc_res[0][0].data, np.array([[0, 1], [0, 0], [1, 1], [1, 0]]).reshape((4, 2)))


def test_keypoints_horizontal_flip_within_stream():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = augs_data.KeyPoints(kpts_data, 2, 2)
    stream = augs_core.Stream([
        trf.RandomFlip(p=1, axis=1)
        ])
    dc = augs_data.DataContainer((kpts, ), 'P')

    dc_res = stream(dc)

    assert np.array_equal(dc_res[0][0].data, np.array([[1, 0], [1, 1], [0, 0], [0, 1]]).reshape((4, 2)))


def test_keypoints_vertical_flip_within_stream():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = augs_data.KeyPoints(kpts_data, 2, 2)
    stream = augs_core.Stream([
        trf.RandomFlip(p=1, axis=0)
        ])
    dc = augs_data.DataContainer((kpts, ), 'P')

    dc_res = stream(dc)

    assert np.array_equal(dc_res[0][0].data, np.array([[0, 1], [0, 0], [1, 1], [1, 0]]).reshape((4, 2)))


def test_rotate_90_img_mask_keypoints(img_mask_3x3):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2))
    kpts = augs_data.KeyPoints(kpts_data, 3, 3)
    img, mask = img_mask_3x3
    H, W = mask.shape

    dc = augs_data.DataContainer((img, mask, kpts,), 'IMP')
    # Defining the 90 degrees transform (clockwise)
    stream = trf.RandomRotate(rotation_range=(90, 90), p=1)
    dc_res = stream(dc)

    img_res, _ = dc_res[0]
    mask_res, _ = dc_res[1]
    kpts_res, _ = dc_res[2]

    M = cv2.getRotationMatrix2D((W // 2, H // 2), -90, 1)
    expected_img_res = cv2.warpAffine(img, M, (W, H)).reshape((H, W, 1))
    expected_mask_res = cv2.warpAffine(mask, M, (W, H))
    expected_kpts_res = np.array([[2, 0], [0, 0], [0, 2], [2, 2]]).reshape((4, 2))

    assert np.array_equal(expected_img_res, img_res)
    assert np.array_equal(expected_mask_res, mask_res)
    np.testing.assert_array_almost_equal(expected_kpts_res, kpts_res.data)


def test_zoom_x_axis_odd(img_5x5):
    stream = trf.RandomScale(range_x=(0.5, 0.5), range_y=(1, 1), same=False, p=1)
    dc = augs_data.DataContainer((img_5x5, ), 'I')
    H, W = img_5x5.shape[0], img_5x5.shape[1]
    img_res = stream(dc)[0][0]
    assert H == img_res.shape[0]
    assert W // 2 == img_res.shape[1]


def test_scale_x_axis_even(img_6x6):
    stream = trf.RandomScale((0.5, 0.5), (1, 1), same=False, p=1)
    dc = augs_data.DataContainer((img_6x6, ), 'I')
    H, W = img_6x6.shape[0], img_6x6.shape[1]
    img_res = stream(dc)[0][0]
    assert H == img_res.shape[0]
    assert W // 2 == img_res.shape[1]


def test_scale_xy_axis_odd(img_5x5):
    stream = trf.RandomScale((0.5, 0.5), (3, 3), same=False, p=1)
    dc = augs_data.DataContainer((img_5x5, ), 'I')
    H, W = img_5x5.shape[0], img_5x5.shape[1]
    img_res = stream(dc)[0][0]
    assert H * 3 == img_res.shape[0]
    assert W // 2 == img_res.shape[1]


def test_scale_xy_axis_even(img_6x6):
    stream = trf.RandomScale((0.5, 0.5), (2, 2), same=False, p=1)
    dc = augs_data.DataContainer((img_6x6, ), 'I')
    H, W = img_6x6.shape[0], img_6x6.shape[1]
    img_res = stream(dc)[0][0]
    assert H * 2 == img_res.shape[0]
    assert W // 2 == img_res.shape[1]


def test_scale_img_mask(img_mask_3x4):
    stream = trf.RandomScale((0.5, 0.5), (2, 2), same=False, p=1)
    dc = augs_data.DataContainer(img_mask_3x4, 'IM')
    H, W = img_mask_3x4[0].shape[0], img_mask_3x4[0].shape[1]
    img_res = stream(dc)[0][0]
    mask_res = stream(dc)[1][0]
    assert H * 2 == img_res.shape[0],  W // 2 == img_res.shape[1]
    assert H * 2 == mask_res.shape[0],  W // 2 == mask_res.shape[1]


def test_keypoints_assert_reflective(img_mask_3x3):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2))
    kpts = augs_data.KeyPoints(kpts_data, 3, 3)
    img, mask = img_mask_3x3

    dc = augs_data.DataContainer((img, mask, kpts,), 'IMP')
    # Defining the 90 degrees transform (clockwise)
    stream = trf.RandomRotate(rotation_range=(20, 20), p=1, padding='r')
    with pytest.raises(ValueError):
        stream(dc)


def test_padding_img_2x2_4x4(img_2x2):
    img = img_2x2
    dc = augs_data.DataContainer((img, ), 'I')
    transf = trf.PadTransform((4, 4))
    res = transf(dc)
    assert (res[0][0].shape[0] == 4) and (res[0][0].shape[1] == 4)


def test_padding_img_2x2_2x2(img_2x2):
    img = img_2x2
    dc = augs_data.DataContainer((img, ), 'I')
    transf = trf.PadTransform((2, 2))
    res = transf(dc)
    assert (res[0][0].shape[0] == 2) and (res[0][0].shape[1] == 2)


def test_padding_img_mask_2x2_4x4(img_mask_2x2):
    img, mask = img_mask_2x2
    dc = augs_data.DataContainer((img, mask), 'IM')
    transf = trf.PadTransform((4, 4))
    res = transf(dc)
    assert (res[0][0].shape[0] == 4) and (res[0][0].shape[1] == 4)
    assert (res[1][0].shape[0] == 4) and (res[1][0].shape[1] == 4)


def test_padding_img_2x2_3x3(img_2x2):
    img = img_2x2
    dc = augs_data.DataContainer((img, ), 'I')
    transf = trf.PadTransform((3, 3))
    res = transf(dc)
    assert (res[0][0].shape[0] == 3) and (res[0][0].shape[1] == 3)


def test_padding_img_mask_2x2_3x3(img_mask_2x2):
    img, mask = img_mask_2x2
    dc = augs_data.DataContainer((img, mask), 'IM')
    transf = trf.PadTransform((3, 3))
    res = transf(dc)
    assert (res[0][0].shape[0] == 3) and (res[0][0].shape[1] == 3)
    assert (res[1][0].shape[0] == 3) and (res[1][0].shape[1] == 3)


def test_padding_img_mask_3x4_3x4(img_mask_3x4):
    img, mask = img_mask_3x4
    dc = augs_data.DataContainer((img, mask), 'IM')
    transf = trf.PadTransform((4, 3))
    res = transf(dc)
    assert (res[0][0].shape[0] == 3) and (res[0][0].shape[1] == 4)
    assert (res[1][0].shape[0] == 3) and (res[1][0].shape[1] == 4)


def test_padding_img_mask_3x4_5x5(img_mask_3x4):
    img, mask = img_mask_3x4
    dc = augs_data.DataContainer((img, mask), 'IM')
    transf = trf.PadTransform((5, 5))
    res = transf(dc)
    assert (res[0][0].shape[0] == 5) and (res[0][0].shape[1] == 5)
    assert (res[1][0].shape[0] == 5) and (res[1][0].shape[1] == 5)


def test_pad_to_20x20_img_mask_keypoints_3x3(img_mask_3x3):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2))
    kpts = augs_data.KeyPoints(kpts_data, 3, 3)
    img, mask = img_mask_3x3

    dc = augs_data.DataContainer((img, mask, kpts,), 'IMP')
    transf = trf.PadTransform((20, 20))
    res = transf(dc)

    assert (res[0][0].shape[0] == 20) and (res[0][0].shape[1] == 20)
    assert (res[1][0].shape[0] == 20) and (res[1][0].shape[1] == 20)
    assert (res[2][0].H == 20) and (res[2][0].W == 20)

    assert np.array_equal(res[2][0].data, np.array([[8, 8], [8, 10], [10, 10], [10, 8]]).reshape((4, 2)))


def test_3x3_pad_to_20x20_center_crop_3x3_shape_stayes_unchanged(img_mask_3x3):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2))
    kpts = augs_data.KeyPoints(kpts_data, 3, 3)
    img, mask = img_mask_3x3

    dc = augs_data.DataContainer((img, mask, kpts,), 'IMP')

    stream = augs_core.Stream([
        trf.PadTransform((20, 20)),
        trf.CropTransform((3, 3))
    ])
    res = stream(dc)

    assert (res[0][0].shape[0] == 3) and (res[0][0].shape[1] == 3)
    assert (res[1][0].shape[0] == 3) and (res[1][0].shape[1] == 3)
    assert (res[2][0].H == 3) and (res[2][0].W == 3)


def test_2x2_pad_to_20x20_center_crop_2x2(img_mask_2x2):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 1], [1, 1], [1, 0]]).reshape((4, 2))
    kpts = augs_data.KeyPoints(kpts_data, 2, 2)
    img, mask = img_mask_2x2

    dc = augs_data.DataContainer((img, mask, kpts,), 'IMP')

    stream = augs_core.Stream([
        trf.PadTransform((20, 20)),
        trf.CropTransform((2, 2))
    ])
    res = stream(dc)

    assert (res[0][0].shape[0] == 2) and (res[0][0].shape[1] == 2)
    assert (res[1][0].shape[0] == 2) and (res[1][0].shape[1] == 2)
    assert (res[2][0].H == 2) and (res[2][0].W == 2)

    assert np.array_equal(res[0][0], img)
    assert np.array_equal(res[1][0], mask)
    assert np.array_equal(res[2][0].data, kpts_data)


def test_6x6_pad_to_20x20_center_crop_6x6(img_6x6):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 5], [1, 3], [2, 0]]).reshape((4, 2))
    kpts = augs_data.KeyPoints(kpts_data, 6, 6)
    img = img_6x6

    dc = augs_data.DataContainer((img,  kpts,), 'IP')

    stream = augs_core.Stream([
        trf.PadTransform((20, 20)),
        trf.CropTransform((6, 6))
    ])
    res = stream(dc)

    assert (res[0][0].shape[0] == 6) and (res[0][0].shape[1] == 6)
    assert (res[1][0].H == 6) and (res[1][0].W == 6)

    assert np.array_equal(res[0][0], img)
    assert np.array_equal(res[1][0].data, kpts_data)
