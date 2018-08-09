import fastaug.core as augs_core
import fastaug.data as augs_data
import fastaug.transforms as imtrf
import numpy as np
import pytest
import cv2

from .fixtures import img_2x2, img_3x4, img_mask_2x2, img_mask_3x4, img_5x5


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


def test_nested_pipeline(img_mask_3x4):
    img, mask = img_mask_3x4
    dc = augs_data.DataContainer((img, mask), 'IM')

    pipeline = augs_core.Pipeline([
        imtrf.RandomFlip(p=1, axis=0),
        imtrf.RandomFlip(p=1, axis=1),
        augs_core.Pipeline([
            imtrf.RandomFlip(p=1, axis=1),
            imtrf.RandomFlip(p=1, axis=0),
        ])
    ])

    dc = pipeline(dc)
    img_res, t0 = dc[0]
    mask_res, t1 = dc[1]

    assert np.array_equal(img, img_res)
    assert np.array_equal(mask, mask_res)


def test_image_shape_equal_3_after_nested_flip(img_3x4):
    img = img_3x4
    dc = augs_data.DataContainer((img, ), 'I')

    pipeline = augs_core.Pipeline([
        imtrf.RandomFlip(p=1, axis=0),
        imtrf.RandomFlip(p=1, axis=1),
        augs_core.Pipeline([
            imtrf.RandomFlip(p=1, axis=1),
            imtrf.RandomFlip(p=1, axis=0),
        ])
    ])

    dc = pipeline(dc)
    img_res, _ = dc[0]

    assert np.array_equal(len(img.shape), 3)


def test_create_empty_keypoints():
    kpts = augs_data.KeyPoints()
    assert kpts.H is None
    assert kpts.W is None
    assert kpts.data is None


def test_create_4_keypoints():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = augs_data.KeyPoints(kpts_data, 3, 4)
    assert kpts.H == 3
    assert kpts.W == 4
    assert np.array_equal(kpts_data, kpts.data)


def test_create_4_keypoints_change_frame():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = augs_data.KeyPoints(kpts_data, 3, 4)
    kpts.H = 2
    kpts.W = 2

    assert kpts.H == 2
    assert kpts.W == 2
    assert np.array_equal(kpts_data, kpts.data)


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


def test_fusion_happens(img_5x5):
    img = img_5x5
    dc = augs_data.DataContainer((img,), 'I')

    ppl = augs_core.Pipeline([
        imtrf.RandomScale((0.5, 1.5), (0.5, 1.5), p=1),
        imtrf.RandomRotate((-50, 50), padding='z', p=1),
        imtrf.RandomShear((-0.5, 0.5), (-0.5, 0.5),padding='z', p=1),
        imtrf.RandomFlip(p=1, axis=1),
    ])

    st = ppl.optimize_stack(ppl.transforms)
    assert len(st) == 2


def test_fusion_rotate_360(img_5x5):
    img = img_5x5
    dc = augs_data.DataContainer((img,), 'I')

    ppl = augs_core.Pipeline([
        imtrf.RandomRotate((45, 45), padding='z', p=1),
        imtrf.RandomRotate((45, 45), padding='z', p=1),
        imtrf.RandomRotate((45, 45), padding='z', p=1),
        imtrf.RandomRotate((45, 45), padding='z', p=1),
        imtrf.RandomRotate((45, 45), padding='z', p=1),
        imtrf.RandomRotate((45, 45), padding='z', p=1),
        imtrf.RandomRotate((45, 45), padding='z', p=1),
        imtrf.RandomRotate((45, 45), padding='z', p=1),
    ])

    img_res = ppl(dc)[0][0]

    np.testing.assert_array_almost_equal(img, img_res)


def test_fusion_rotate_360_flip_rotate_360(img_5x5):
    img = img_5x5
    dc = augs_data.DataContainer((img,), 'I')

    ppl = augs_core.Pipeline([
        imtrf.RandomRotate((45, 45), padding='z', p=1),
        imtrf.RandomRotate((45, 45), padding='z', p=1),
        imtrf.RandomRotate((45, 45), padding='z', p=1),
        imtrf.RandomRotate((45, 45), padding='z', p=1),
        imtrf.RandomRotate((45, 45), padding='z', p=1),
        imtrf.RandomRotate((45, 45), padding='z', p=1),
        imtrf.RandomRotate((45, 45), padding='z', p=1),
        imtrf.RandomRotate((45, 45), padding='z', p=1),
        imtrf.RandomFlip(p=1, axis=1),
        augs_core.Pipeline([
            imtrf.RandomRotate((45, 45), padding='z', p=1),
            imtrf.RandomRotate((45, 45), padding='z', p=1),
            imtrf.RandomRotate((45, 45), padding='z', p=1),
            imtrf.RandomRotate((45, 45), padding='z', p=1),
            imtrf.RandomRotate((45, 45), padding='z', p=1),
            imtrf.RandomRotate((45, 45), padding='z', p=1),
            imtrf.RandomRotate((45, 45), padding='z', p=1),
            imtrf.RandomRotate((45, 45), padding='z', p=1),
        ])
    ])

    img_res = ppl(dc)[0][0]

    np.testing.assert_array_almost_equal(cv2.flip(img, 1).reshape(5, 5, 1), img_res)