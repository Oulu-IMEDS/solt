import fastaug.core as augs_core
import fastaug.data as augs_data
import fastaug.transforms as trf
import numpy as np
import cv2

import pytest

_globals = {'img': None, 'mask':None}


@pytest.fixture
def img_2x2():
    """
    Generates a 2x2 grayscale image (uint8)

    Returns
    -------
    out : ndarray
        2x2x1 uint8 image
    """
    return np.array([[1, 0], [1, 1]]).reshape((2, 2, 1)).astype(np.uint8)


@pytest.fixture
def img_mask_2x2():
    """
    Generates 2x2 mask (doesn't have the 3rd dimension compare to an image).

    Returns
    -------
    out : ndarray
        2x2 mask, uint8
    """
    return img_2x2(), np.array([[1, 0], [0, 1]]).reshape((2, 2)).astype(np.uint8) # Generating the mask as well


@pytest.fixture
def img_3x4():
    """
    Generates a grayscale image 3x4

    Returns
    -------
    out : ndarray
        3x4x1 uint8 image
    """
    img = np.array([[1, 1, 1, 0],
                    [1, 0, 1, 1],
                    [1, 1, 1, 1]
                    ]).reshape((3, 4, 1)).astype(np.uint8)
    return img


@pytest.fixture
def img_mask_3x4():
    """
    Generates a mask  3x4

    Returns
    -------
    out : ndarray
        3x4 uint8 image
    """
    img = img_3x4()
    mask = np.array([[0, 1, 1, 1],
                    [0, 1, 1, 0],
                    [0, 1, 1, 0]
                    ]).reshape((3, 4)).astype(np.uint8)
    return img, mask


def test_data_item_create_img(img_2x2):
    img = img_2x2
    dc = augs_data.DataContainer((img,), 'I')
    assert len(dc) == 1
    assert np.array_equal(img, dc[0][0])
    assert dc[0][1] == 'I'


def test_pipeline_empty(img_2x2):
    img = img_2x2
    dc = augs_data.DataContainer((img,), 'I')
    pipeline = augs_core.Pipeline()
    res, _ = pipeline(dc)[0]
    assert np.all(res == img)


def test_empty_pipeline_selective():
    with pytest.raises(AssertionError):
        augs_core.SelectivePipeline()



def test_img_mask_horizontal_flip(img_mask_3x4):
    img, mask = img_mask_3x4
    dc = augs_data.DataContainer((img, mask), 'IM')

    pipeline = augs_core.Pipeline([
        trf.RandomFlip(p=1)
    ])

    dc = pipeline(dc)
    img_res, _ = dc[0]
    mask_res, _ = dc[1]

    h, w = mask.shape
    assert np.array_equal(cv2.flip(img, 0).reshape(h, w, 1), img_res)
    assert np.array_equal(cv2.flip(mask, 0), mask_res)


def test_img_mask_mask_horizontal_flip(img_mask_3x4):
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


def test_img_mask_vertical_flip(img_mask_3x4):
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



def test_img_mask_hoizontal_vertical_flip(img_mask_3x4):
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


def test_image_shape_equal_3_after_nested_flip(img_3x4):
    img = img_3x4
    dc = augs_data.DataContainer((img), 'I')

    pipeline = augs_core.Pipeline([
        trf.RandomFlip(p=1, axis=0),
        trf.RandomFlip(p=1, axis=1),
        augs_core.Pipeline([
            trf.RandomFlip(p=1, axis=1),
            trf.RandomFlip(p=1, axis=0),
        ])
    ])

    dc = pipeline(dc)
    img_res, _ = dc[0]

    assert np.array_equal(len(img.shape), 3)


def test_nested_pipeline(img_mask_3x4):
    img, mask = img_mask_3x4
    dc = augs_data.DataContainer((img, mask), 'IM')

    pipeline = augs_core.Pipeline([
        trf.RandomFlip(p=1, axis=0),
        trf.RandomFlip(p=1, axis=1),
        augs_core.Pipeline([
            trf.RandomFlip(p=1, axis=1),
            trf.RandomFlip(p=1, axis=0),
        ])
    ])

    dc = pipeline(dc)
    img_res, t0 = dc[0]
    mask_res, t1 = dc[1]

    assert np.array_equal(img, img_res)
    assert np.array_equal(mask, mask_res)


def test_create_empty_keypoints():
    kpts = augs_data.KeyPoints()
    assert kpts.H is None
    assert kpts.W is None
    assert kpts.pts is None


def test_create_4_keypoints():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = augs_data.KeyPoints(kpts_data, 3, 4)
    assert kpts.H == 3
    assert kpts.W == 4
    assert np.array_equal(kpts_data, kpts.pts)


def test_create_4_keypoints_change_frame():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = augs_data.KeyPoints(kpts_data, 3, 4)
    kpts.H = 2
    kpts.W = 2

    assert kpts.H == 2
    assert kpts.W == 2
    assert np.array_equal(kpts_data, kpts.pts)


def test_create_4_keypoints_change_grid_and_frame():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = augs_data.KeyPoints(kpts_data, 3, 4)

    kpts_data_new = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5]]).reshape((5, 2))
    kpts.H = 2
    kpts.W = 2
    kpts.pts = kpts_data_new

    assert kpts.H == 2
    assert kpts.W == 2
    assert np.array_equal(kpts_data_new, kpts.pts)
