import fastaug.core as augs_core
import fastaug.data as augs_data
import fastaug.transforms as trf
import numpy as np
import cv2
import pytest

from .fixtures import img_2x2, img_3x4, img_mask_2x2, img_mask_3x4

def test_img_mask_horizontal_flip(img_mask_3x4):
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


def test_keypoints_horizontal_flip():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = augs_data.KeyPoints(kpts_data, 2, 2)
    pipeline = trf.RandomFlip(p=1, axis=0)
    dc = augs_data.DataContainer((kpts, ), 'P')

    dc_res = pipeline(dc)

    assert np.array_equal(dc_res[0][0].data, np.array([[1, 0], [1, 1], [0, 0], [0, 1]]).reshape((4, 2)))


def test_keypoints_vertical_flip():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = augs_data.KeyPoints(kpts_data, 2, 2)
    pipeline = trf.RandomFlip(p=1, axis=1)
    dc = augs_data.DataContainer((kpts, ), 'P')

    dc_res = pipeline(dc)

    assert np.array_equal(dc_res[0][0].data, np.array([[0, 1], [0, 0], [1, 1], [1, 0]]).reshape((4, 2)))
