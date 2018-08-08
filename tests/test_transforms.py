import fastaug.core as augs_core
import fastaug.data as augs_data
import fastaug.transforms as trf
import numpy as np
import cv2
import pytest

from .fixtures import img_2x2, img_3x4, img_mask_2x2, img_mask_3x4, img_mask_3x3, img_5x5, img_6x6


def test_img_mask_vertical_flip(img_mask_3x4):
    img, mask = img_mask_3x4
    dc = augs_data.DataContainer((img, mask), 'IM')

    pipeline = augs_core.Pipeline([
        trf.RandomFlip(p=1, axis=0)
    ])

    dc = pipeline(dc)
    img_res, _ = dc[0]
    mask_res, _ = dc[1]

    h, w = mask.shape
    assert np.array_equal(cv2.flip(img, 0).reshape(h, w, 1), img_res)
    assert np.array_equal(cv2.flip(mask, 0), mask_res)


def test_img_mask_mask_vertical_flip(img_mask_3x4):
    img, mask = img_mask_3x4
    dc = augs_data.DataContainer((img, mask, mask), 'IMM')

    pipeline = augs_core.Pipeline([
        trf.RandomFlip(p=1, axis=0)
    ])

    dc = pipeline(dc)
    img_res, _ = dc[0]
    mask_res, _ = dc[1]

    h, w = mask.shape
    assert np.array_equal(cv2.flip(img, 0).reshape(h, w, 1), img_res)
    assert np.array_equal(cv2.flip(mask, 0), mask_res)


def test_img_mask_horizontal_flip(img_mask_3x4):
    img, mask = img_mask_3x4
    dc = augs_data.DataContainer((img, mask), 'IM')

    pipeline = augs_core.Pipeline([
        trf.RandomFlip(p=1, axis=1)
    ])

    dc = pipeline(dc)
    img_res, _ = dc[0]
    mask_res, _ = dc[1]

    h, w = mask.shape
    assert np.array_equal(cv2.flip(img, 1).reshape(h, w, 1), img_res)
    assert np.array_equal(cv2.flip(mask, 1), mask_res)


def test_img_mask_vertical_horizontal_flip(img_mask_3x4):
    img, mask = img_mask_3x4
    dc = augs_data.DataContainer((img, mask), 'IM')

    pipeline = augs_core.Pipeline([
        trf.RandomFlip(p=1, axis=0),
        trf.RandomFlip(p=1, axis=1)
    ])

    dc = pipeline(dc)
    img_res, _ = dc[0]
    mask_res, _ = dc[1]

    h, w = mask.shape
    assert np.array_equal(cv2.flip(cv2.flip(img, 0), 1).reshape(h, w, 1), img_res)
    assert np.array_equal(cv2.flip(cv2.flip(mask, 0), 1), mask_res)


def test_keypoints_vertical_flip():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = augs_data.KeyPoints(kpts_data, 2, 2)
    pipeline = trf.RandomFlip(p=1, axis=0)
    dc = augs_data.DataContainer((kpts, ), 'P')

    dc_res = pipeline(dc)

    assert np.array_equal(dc_res[0][0].data, np.array([[1, 0], [1, 1], [0, 0], [0, 1]]).reshape((4, 2)))


def test_keypoints_horizontal_flip_within_pipeline():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = augs_data.KeyPoints(kpts_data, 2, 2)
    pipeline = augs_core.Pipeline([
        trf.RandomFlip(p=1, axis=0)
        ])
    dc = augs_data.DataContainer((kpts, ), 'P')

    dc_res = pipeline(dc)

    assert np.array_equal(dc_res[0][0].data, np.array([[1, 0], [1, 1], [0, 0], [0, 1]]).reshape((4, 2)))


def test_keypoints_vertical_flip():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = augs_data.KeyPoints(kpts_data, 2, 2)
    pipeline = augs_core.Pipeline([
        trf.RandomFlip(p=1, axis=1)
        ])
    dc = augs_data.DataContainer((kpts, ), 'P')

    dc_res = pipeline(dc)

    assert np.array_equal(dc_res[0][0].data, np.array([[0, 1], [0, 0], [1, 1], [1, 0]]).reshape((4, 2)))


def test_rotate_90_img_mask_keypoints(img_mask_3x3):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2))
    kpts = augs_data.KeyPoints(kpts_data, 3, 3)
    img, mask = img_mask_3x3
    H, W = mask.shape

    dc = augs_data.DataContainer((img, mask, kpts,), 'IMP')
    # Defining the 90 degrees transform (clockwise)
    pipeline = trf.RandomRotate(rotation_range=(90, 90), p=1)
    dc_res = pipeline(dc)

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
    pipeline = trf.RandomScale((0.5, 0.5), (1, 1), p=1)
    dc = augs_data.DataContainer((img_5x5, ), 'I')
    H, W = img_5x5.shape[0], img_5x5.shape[1]
    img_res = pipeline(dc)[0][0]
    assert img_res.shape[0] == H, img_res.shape[1] == W // 2


def test_scale_x_axis_even(img_6x6):
    pipeline = trf.RandomScale((0.5, 0.5), (1, 1), p=1)
    dc = augs_data.DataContainer((img_6x6, ), 'I')
    H, W = img_6x6.shape[0], img_6x6.shape[1]
    img_res = pipeline(dc)[0][0]
    assert img_res.shape[0] == H, img_res.shape[1] == W // 2


def test_scale_xy_axis_odd(img_5x5):
    pipeline = trf.RandomScale((0.5, 0.5), (3, 3), p=1)
    dc = augs_data.DataContainer((img_5x5, ), 'I')
    H, W = img_5x5.shape[0], img_5x5.shape[1]
    img_res = pipeline(dc)[0][0]
    assert H * 3 == img_res.shape[0], W // 2 == img_res.shape[1]


def test_scale_x_axis_even(img_6x6):
    pipeline = trf.RandomScale((0.5, 0.5), (2, 2), p=1)
    dc = augs_data.DataContainer((img_6x6, ), 'I')
    H, W = img_6x6.shape[0], img_6x6.shape[1]
    img_res = pipeline(dc)[0][0]
    assert H * 2 == img_res.shape[0],  W // 2 == img_res.shape[1]


def test_scale_img_mask(img_mask_3x4):
    pipeline = trf.RandomScale((0.5, 0.5), (2, 2), p=1)
    dc = augs_data.DataContainer(img_mask_3x4, 'IM')
    H, W = img_mask_3x4[0].shape[0], img_mask_3x4[0].shape[1]
    img_res = pipeline(dc)[0][0]
    mask_res = pipeline(dc)[1][0]
    assert H * 2 == img_res.shape[0],  W // 2 == img_res.shape[1]
    assert H * 2 == mask_res.shape[0],  W // 2 == mask_res.shape[1]


def test_keypoints_assert_reflective(img_mask_3x3):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2))
    kpts = augs_data.KeyPoints(kpts_data, 3, 3)
    img, mask = img_mask_3x3
    H, W = mask.shape

    dc = augs_data.DataContainer((img, mask, kpts,), 'IMP')
    # Defining the 90 degrees transform (clockwise)
    pipeline = trf.RandomRotate(rotation_range=(20, 20), p=1, padding='reflect')
    with pytest.raises(ValueError):
        pipeline(dc)