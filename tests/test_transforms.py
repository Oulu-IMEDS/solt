import copy
import random
from contextlib import ExitStack as does_not_raise

import cv2
import numpy as np
import torch
import pytest

import solt.core as slc

import solt.transforms as slt
from solt.constants import ALLOWED_INTERPOLATIONS, ALLOWED_PADDINGS

from .fixtures import *


@pytest.mark.parametrize("img, mask", [(img_3x4(), mask_3x4())])
def test_img_mask_vertical_flip(img, mask):
    dc = slc.DataContainer((img, mask), "IM")

    stream = slc.Stream([slt.Flip(p=1, axis=0),])

    dc = stream(dc, return_torch=False)
    img_res, _, _ = dc[0]
    mask_res, _, _ = dc[1]

    h, w = mask.shape
    assert np.array_equal(cv2.flip(img, 0).reshape(h, w, 1), img_res)
    assert np.array_equal(cv2.flip(mask, 0), mask_res)


@pytest.mark.parametrize("img, mask", [(img_3x4(), mask_3x4())])
def test_img_mask_mask_vertical_flip(img, mask):
    dc = slc.DataContainer((img, mask, mask), "IMM")

    stream = slc.Stream([slt.Flip(p=1, axis=0)])

    dc = stream(dc, return_torch=False)
    img_res, _, _ = dc[0]
    mask_res, _, _ = dc[1]

    h, w = mask.shape
    assert np.array_equal(cv2.flip(img, 0).reshape(h, w, 1), img_res)
    assert np.array_equal(cv2.flip(mask, 0), mask_res)


@pytest.mark.parametrize("img, mask", [(img_3x4(), mask_3x4())])
def test_img_mask_horizontal_flip(img, mask):
    dc = slc.DataContainer((img, mask), "IM")

    stream = slc.Stream([slt.Flip(p=1, axis=1)])

    dc = stream(dc, return_torch=False)
    img_res, _, _ = dc[0]
    mask_res, _, _ = dc[1]

    h, w = mask.shape
    assert np.array_equal(cv2.flip(img, 1).reshape(h, w, 1), img_res)
    assert np.array_equal(cv2.flip(mask, 1), mask_res)


@pytest.mark.parametrize("img, mask", [(img_3x4(), mask_3x4())])
def test_img_mask_vertical_horizontal_flip(img, mask):
    dc = slc.DataContainer((img, mask), "IM")

    stream = slc.Stream([slt.Flip(p=1, axis=0), slt.Flip(p=1, axis=1)])

    dc = stream(dc, return_torch=False)
    img_res, _, _ = dc[0]
    mask_res, _, _ = dc[1]

    h, w = mask.shape
    assert np.array_equal(cv2.flip(cv2.flip(img, 0), 1).reshape(h, w, 1), img_res)
    assert np.array_equal(cv2.flip(cv2.flip(mask, 0), 1), mask_res)


@pytest.mark.parametrize("img, mask", [(img_3x4(), mask_3x4())])
def test_img_mask_kpts_flip_all_axes(img, mask):
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    kpts = slc.Keypoints(kpts_data.copy(), frame=(3, 4))
    dc = slc.DataContainer((img, mask, kpts), "IMP")

    stream = slt.Flip(p=1, axis=(0, 1))

    dc = stream(dc)
    img_res, _, _ = dc[0]
    mask_res, _, _ = dc[1]
    kpts_res, _, _ = dc[2]

    h, w = mask.shape
    assert np.array_equal(cv2.flip(cv2.flip(img, 0), 1).reshape(h, w, 1), img_res)
    assert np.array_equal(cv2.flip(cv2.flip(mask, 0), 1), mask_res)

    kpts_data[:, 0] = (3 - 1) - kpts_data[:, 0]
    kpts_data[:, 1] = (4 - 1) - kpts_data[:, 1]

    assert np.array_equal(kpts_data, kpts_res.data)


def test_keypoints_flip():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, frame=(2, 2))
    stream = slt.Flip(p=1, axis=0)
    dc = slc.DataContainer((kpts,), "P")

    dc_res = stream(dc)

    expected = np.array([[1, 0], [1, 1], [0, 0], [0, 1]]).reshape((4, 2))
    assert np.array_equal(dc_res[0][0].data, expected)


@pytest.mark.parametrize(
    "axis, kpts_expected",
    [(0, np.array([[1, 0], [1, 1], [0, 0], [0, 1]])), (1, np.array([[0, 1], [0, 0], [1, 1], [1, 0]]))],
)
def test_keypoints_flip_within_stream(axis, kpts_expected):
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, frame=(2, 2))

    stream = slc.Stream([slt.Flip(p=1, axis=axis)])
    dc = slc.DataContainer((kpts,), "P")
    dc_res = stream(dc, return_torch=False)
    assert np.array_equal(dc_res[0][0].data, kpts_expected)


def test_rotate_range_none():
    trf = slt.Rotate(None)
    assert trf.angle_range == (0, 0)


@pytest.mark.parametrize("angle", [1, 2.5])
def test_rotate_range_conversion_from_number(angle):
    trf = slt.Rotate(angle_range=angle)
    assert trf.angle_range == (-angle, angle)


def test_shear_range_none():
    trf = slt.Shear(None, None)
    assert trf.range_x == (0, 0)
    assert trf.range_y == (0, 0)


@pytest.mark.parametrize("img, mask", [(img_3x3(), mask_3x3())])
@pytest.mark.parametrize("ignore_state", [True, False])
@pytest.mark.parametrize(
    "transform_settings",
    [
        None,
        {0: {"interpolation": "nearest", "padding": "z"}},
        {0: {"interpolation": "nearest", "padding": "r"}},
        {0: {"interpolation": "bilinear", "padding": "z"}},
        {0: {"interpolation": "bilinear", "padding": "r"}},
        {0: {"interpolation": "bicubic", "padding": "z"}},
        {0: {"interpolation": "bicubic", "padding": "r"}},
        {0: {"interpolation": "area", "padding": "z"}},
        {0: {"interpolation": "area", "padding": "r"}},
        {0: {"interpolation": "lanczos", "padding": "z"}},
        {0: {"interpolation": "lanczos", "padding": "r"}},
    ],
)
def test_rotate_90_img_mask_keypoints_destructive(img, mask, transform_settings, ignore_state):
    # Setting up the data

    kpts_data = np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, frame=(3, 3))
    H, W = mask.shape

    dc = slc.DataContainer((img, mask, kpts, 1), "IMPL", transform_settings=copy.deepcopy(transform_settings),)
    # Defining the 90 degrees transform (clockwise)
    stream = slt.Rotate(angle_range=(90, 90), p=1, ignore_state=ignore_state)
    dc_res = stream(dc)

    img_res, _, _ = dc_res[0]
    mask_res, _, _ = dc_res[1]
    kpts_res, _, _ = dc_res[2]
    label_res, _, _ = dc_res[3]
    M = cv2.getRotationMatrix2D((W // 2, H // 2), -90, 1)

    img_inter = ALLOWED_INTERPOLATIONS["bicubic"]
    img_pad = ALLOWED_PADDINGS["z"]
    if transform_settings is not None:
        img_inter = ALLOWED_INTERPOLATIONS[transform_settings[0]["interpolation"]]
        img_pad = ALLOWED_PADDINGS[transform_settings[0]["padding"]]

    expected_img_res = cv2.warpAffine(img, M, (W, H), flags=img_inter, borderMode=img_pad).reshape((H, W, 1))
    expected_mask_res = cv2.warpAffine(mask, M, (W, H))
    expected_kpts_res = np.array([[2, 0], [0, 0], [0, 2], [2, 2]]).reshape((4, 2))

    assert np.array_equal(expected_img_res, img_res)
    assert np.array_equal(expected_mask_res, mask_res)
    np.testing.assert_array_almost_equal(expected_kpts_res, kpts_res.data)
    assert label_res == 1


@pytest.mark.parametrize("img, mask", [(img_3x3(), mask_3x3())])
@pytest.mark.parametrize("k", list(range(-4, 5)))
def test_rotate_90_img_mask_nondestructive(k, img, mask):
    # Setting up the data
    H, W = mask.shape

    dc = slc.DataContainer((img, mask), "IM")
    # Defining the 90 degrees transform (counterclockwise)
    stream = slt.Rotate90(k=k, p=1)
    dc_res = stream(dc)

    img_res, _, _ = dc_res[0]
    mask_res, _, _ = dc_res[1]

    expected_img_res = np.rot90(img, -k).reshape((H, W, 1))
    expected_mask_res = np.rot90(mask, -k)

    assert np.array_equal(expected_img_res, img_res)
    assert np.array_equal(expected_mask_res, mask_res)


@pytest.mark.parametrize("k", [None, "123", 123.0])
def test_rotate_nondestructive_does_not_accept_non_int_k(k):
    with pytest.raises(TypeError):
        slt.Rotate90(k=k)


@pytest.mark.parametrize("img", [img_6x6_rgb(),])
@pytest.mark.parametrize("k", list(range(-4, 5)))
def test_rotate_90_transforms_have_same_behaviour(k, img):
    dc = slc.DataContainer(img, "I")
    trf_1 = slt.Rotate(angle_range=(k * 90, k * 90), p=1)
    trf_1.sample_transform(dc)

    trf_2 = slt.Rotate90(k=k, p=1)
    trf_2.sample_transform(dc)

    assert np.array_equal(trf_1.state_dict["transform_matrix"], trf_2.state_dict["transform_matrix"])


@pytest.mark.parametrize("img", [img_5x5(),])
def test_zoom_x_axis_odd(img):
    stream = slt.Scale(range_x=(0.5, 0.5), range_y=(1, 1), same=False, p=1, ignore_fast_mode=True)
    dc = slc.DataContainer((img,), "I")
    H, W = img.shape[:2]
    img_res = stream(dc)[0][0]
    assert H == img_res.shape[0]
    assert W // 2 == img_res.shape[1]


@pytest.mark.parametrize("img", [img_6x6(),])
def test_scale_x_axis_even(img):
    stream = slt.Scale((0.5, 0.5), (1, 1), same=False, p=1, ignore_fast_mode=True)
    dc = slc.DataContainer((img,), "I")
    H, W = img.shape[:2]
    img_res = stream(dc)[0][0]
    assert H == img_res.shape[0]
    assert W // 2 == img_res.shape[1]


@pytest.mark.parametrize("img", [img_5x5(),])
def test_scale_xy_axis_odd(img):
    stream = slt.Scale((0.5, 0.5), (3, 3), same=False, p=1, ignore_fast_mode=True)
    dc = slc.DataContainer((img,), "I")
    H, W = img.shape[:2]
    img_res = stream(dc)[0][0]
    assert H * 3 == img_res.shape[0]
    assert W // 2 == img_res.shape[1]


@pytest.mark.parametrize("img", [img_6x6(),])
def test_scale_xy_axis_even(img):
    stream = slt.Scale((0.5, 0.5), (2, 2), same=False, p=1, ignore_fast_mode=True)
    dc = slc.DataContainer((img,), "I")
    H, W = img.shape[:2]
    img_res = stream(dc)[0][0]
    assert H * 2 == img_res.shape[0]
    assert W // 2 == img_res.shape[1]


@pytest.mark.parametrize("img, mask", [(img_3x4(), mask_3x4())])
def test_scale_img_mask(img, mask):
    stream = slt.Scale((0.5, 0.5), (2, 2), same=False, p=1, ignore_fast_mode=True)
    dc = slc.DataContainer((img, mask), "IM")
    H, W = img.shape[:2]
    img_res = stream(dc)[0][0]
    mask_res = stream(dc)[1][0]
    assert H * 2 == img_res.shape[0], W // 2 == img_res.shape[1]
    assert H * 2 == mask_res.shape[0], W // 2 == mask_res.shape[1]


@pytest.mark.parametrize("img, mask", [(img_3x3(), mask_3x3())])
def test_keypoints_assert_reflective(img, mask):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, frame=(3, 3))

    dc = slc.DataContainer((img, mask, kpts,), "IMP")
    # Defining the 90 degrees transform (clockwise)
    stream = slt.Rotate(angle_range=(20, 20), p=1, padding="r")
    with pytest.raises(ValueError):
        stream(dc)


@pytest.mark.parametrize("img, pad_to", [(img_2x2(), (2, 2)), (img_2x2(), (3, 3)), (img_2x2(), (4, 4)),])
def test_padding_img(img, pad_to):
    dc = slc.DataContainer((img,), "I")
    transf = slt.Pad(pad_to)
    res = transf(dc)
    np.testing.assert_array_equal(res[0][0].shape[:-1], pad_to)


@pytest.mark.parametrize(
    "img, mask, pad_to, shape_out",
    [
        (img_2x2(), mask_2x2(), (3, 3), (3, 3)),
        (img_2x2(), mask_2x2(), (4, 4), (4, 4)),
        (img_3x4(), mask_3x4(), (5, 5), (5, 5)),
        (img_3x4(), mask_3x4(), (4, 3), (4, 4)),
        (img_5x5(), mask_5x5(), (2, 2), (5, 5)),
    ],
)
def test_padding_img_mask(img, mask, pad_to, shape_out):
    dc = slc.DataContainer((img, mask), "IM")
    transf = slt.Pad(pad_to)
    res = transf(dc)
    np.testing.assert_array_equal(res[0][0].shape[:-1], shape_out)
    np.testing.assert_array_equal(res[1][0].shape, shape_out)


@pytest.mark.parametrize("img, mask", [(img_3x3(), mask_3x3())])
def test_pad_to_20x20_img_mask_keypoints_3x3(img, mask):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 2], [2, 2], [2, 0]])
    kpts = slc.Keypoints(kpts_data, frame=(3, 3))

    dc = slc.DataContainer((img, mask, kpts,), "IMP")
    transf = slt.Pad((20, 20))
    res = transf(dc)

    assert (res[0][0].shape[0] == 20) and (res[0][0].shape[1] == 20)
    assert (res[1][0].shape[0] == 20) and (res[1][0].shape[1] == 20)
    assert (res[2][0].frame[0] == 20) and (res[2][0].frame[1] == 20)

    assert np.array_equal(res[2][0].data, np.array([[8, 8], [8, 10], [10, 10], [10, 8]]))


@pytest.mark.parametrize("img, mask", [(img_3x3(), mask_3x3())])
@pytest.mark.parametrize("trf", [slt.Pad, slt.Crop, slt.Resize])
def test_pad_crop_resize_dont_change_data_when_parameters_are_not_set(img, mask, trf):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 2], [2, 2], [2, 0]])
    kpts = slc.Keypoints(kpts_data, frame=(3, 3))

    dc = slc.DataContainer((img, mask, kpts,), "IMP")
    res = trf()(dc, return_torch=False)
    assert dc == res


@pytest.mark.parametrize(
    "img, mask, resize_to",
    [
        (img_6x6(), mask_6x6(), (20, 20)),
        (img_6x6(), mask_6x6(), (3, 3)),
        (img_6x6(), mask_6x6(), 3),
        (img_6x6(), mask_6x6(), (5, 5)),
        (img_6x6(), mask_6x6(), (4, 4)),
        (img_6x6(), mask_6x6(), (7, 6)),
        (img_6x6(), mask_6x6(), (5, 7)),
        (img_6x6(), mask_6x6(), (6, 6)),
        (img_6x6(), mask_6x6(), (2, 3)),
        (img_5x5(), mask_5x5(), (20, 20)),
        (img_5x5(), mask_5x5(), (3, 3)),
        (img_5x5(), mask_5x5(), (5, 5)),
        (img_5x5(), mask_5x5(), (4, 4)),
        (img_5x5(), mask_5x5(), (7, 6)),
        (img_5x5(), mask_5x5(), (5, 7)),
        (img_5x5(), mask_5x5(), (6, 6)),
        (img_5x5(), mask_5x5(), (2, 3)),
    ],
)
def test_resize_img_to_arbitrary_size(img, mask, resize_to):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2))  # Top left corner
    kpts = slc.Keypoints(kpts_data.copy(), frame=(img.shape[:2]))

    dc = slc.DataContainer((img, mask, kpts,), "IMP")
    transf = slt.Resize(resize_to=resize_to)
    res = transf(dc).data
    if isinstance(resize_to, int):
        resize_to = (resize_to, resize_to)

    scales = tuple(resize_to[i] / img.shape[i] for i in range(img.ndim - 1))

    assert transf.resize_to == resize_to
    np.testing.assert_array_equal(res[0].shape[:-1], resize_to)
    np.testing.assert_array_equal(res[1].shape, resize_to)
    np.testing.assert_array_equal(res[2].frame, resize_to)

    kpts_data = kpts_data.astype(float)
    kpts_data = (
        kpts_data * np.asarray(scales)[None,]
    )
    kpts_data = kpts_data.astype(int)
    assert np.array_equal(res[2].data, kpts_data)


@pytest.mark.parametrize("resize_to", ["1123", [123, 123], 123.0])
def test_wrong_resize_types(resize_to):
    with pytest.raises(TypeError):
        slt.Resize(resize_to=resize_to)


def test_resize_does_not_change_labels():
    trf = slt.Resize(resize_to=(5, 5))
    dc = slc.DataContainer((1,), "L")
    dc = trf(dc)
    assert dc.data[0] == 1


@pytest.mark.parametrize("img, mask", [(img_3x3(), mask_3x3())])
def test_pad_to_20x20_img_mask_keypoints_3x3_kpts_first(img, mask):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, frame=(3, 3))

    dc = slc.DataContainer((kpts, img, mask), "PIM")
    transf = slt.Pad((20, 20))
    res = transf(dc)

    assert (res[2][0].shape[0] == 20) and (res[2][0].shape[1] == 20)
    assert (res[1][0].shape[0] == 20) and (res[1][0].shape[1] == 20)
    assert (res[0][0].frame[0] == 20) and (res[0][0].frame[1] == 20)

    assert np.array_equal(res[0][0].data, np.array([[8, 8], [8, 10], [10, 10], [10, 8]]).reshape((4, 2)))


@pytest.mark.parametrize("img, mask", [(img_3x3(), mask_3x3())])
def test_3x3_pad_to_20x20_center_crop_3x3_shape_stayes_unchanged(img, mask):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, frame=(3, 3))

    dc = slc.DataContainer((img, mask, kpts,), "IMP")

    stream = slc.Stream([slt.Pad((20, 20)), slt.Crop((3, 3))])
    res = stream(dc, return_torch=False)

    assert (res[0][0].shape[0] == 3) and (res[0][0].shape[1] == 3)
    assert (res[1][0].shape[0] == 3) and (res[1][0].shape[1] == 3)
    assert (res[2][0].frame[0] == 3) and (res[2][0].frame[1] == 3)


@pytest.mark.parametrize("pad_size,crop_size", [(20, 2), (20, (2, 2)), ((20, 20), (2, 2)), ((20, 20), 2)])
@pytest.mark.parametrize("img, mask", [(img_2x2(), mask_2x2())])
def test_2x2_pad_to_20x20_center_crop_2x2(pad_size, crop_size, img, mask):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 1], [1, 1], [1, 0]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, frame=(2, 2))

    dc = slc.DataContainer((img, mask, kpts,), "IMP")

    stream = slc.Stream([slt.Pad(pad_to=pad_size), slt.Crop(crop_to=crop_size)])
    res = stream(dc, return_torch=False)

    assert (res[0][0].shape[0] == 2) and (res[0][0].shape[1] == 2)
    assert (res[1][0].shape[0] == 2) and (res[1][0].shape[1] == 2)
    assert res[2][0].frame == (2, 2)

    assert np.array_equal(res[0][0], img)
    assert np.array_equal(res[1][0], mask)
    assert np.array_equal(res[2][0].data, kpts_data)


@pytest.mark.parametrize("crop_mode", ["c", "r", "d"])
@pytest.mark.parametrize(
    "img, mask",
    [
        (img_2x2(), mask_2x2()),
        (torch.ones((3, 4, 5, 1)), torch.ones((3, 4, 5))),
        (torch.ones((3, 3, 7, 4, 6, 3)), torch.zeros((3, 3, 7, 4, 6))),
    ],
)
def test_different_crop_modes(crop_mode, img, mask):
    if crop_mode == "d":
        with pytest.raises(ValueError):
            slt.Crop(crop_to=2, crop_mode=crop_mode)
    else:
        stream = slc.Stream([slt.Pad(pad_to=20), slt.Crop(crop_to=2, crop_mode=crop_mode)])
        dc = slc.DataContainer((img, mask,), "IM")
        dc_res = stream(dc, return_torch=False)

        for el in dc_res.data:
            assert el.shape[0] == 2
            assert el.shape[1] == 2


@pytest.mark.parametrize("img", [img_6x6(),])
def test_6x6_pad_to_20x20_center_crop_6x6_img_kpts(img):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 5], [1, 3], [2, 0]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, frame=(6, 6))

    dc = slc.DataContainer((img, kpts, 1), "IPL")

    stream = slc.Stream([slt.Pad((20, 20)), slt.Crop((6, 6))])
    res = stream(dc, return_torch=False)

    assert (res[0][0].shape[0] == 6) and (res[0][0].shape[1] == 6)
    assert (res[1][0].frame[0] == 6) and (res[1][0].frame[1] == 6)

    assert np.array_equal(res[0][0], img)
    assert np.array_equal(res[1][0].data, kpts_data)


@pytest.mark.parametrize("img", [img_6x6(),])
def test_6x6_pad_to_20x20_center_crop_6x6_kpts_img(img):
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 5], [1, 3], [2, 0]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, frame=(6, 6))

    dc = slc.DataContainer((kpts, img), "PI")

    stream = slc.Stream([slt.Pad((20, 20)), slt.Crop((6, 6))])
    res = stream(dc, return_torch=False)

    assert (res[1][0].shape[0] == 6) and (res[1][0].shape[1] == 6)
    assert (res[0][0].frame[0] == 6) and (res[0][0].frame[1] == 6)

    assert np.array_equal(res[1][0], img)
    assert np.array_equal(res[0][0].data, kpts_data)


@pytest.mark.parametrize("img", [img_6x6_rgb(),])
def test_translate_forward_backward_sampling(img):
    stream = slc.Stream(
        [slt.Translate(range_x=(1, 1), range_y=(1, 1), p=1), slt.Translate(range_x=(-1, -1), range_y=(-1, -1), p=1)]
    )
    dc = slc.DataContainer(img, "I")
    trf = stream.optimize_transforms_stack(stream.transforms, dc)[0]
    assert 1 == trf.state_dict["translate_x"]  # The settings will be overrided by the first transform
    assert 1 == trf.state_dict["translate_y"]  # The settings will be overrided by the first transform
    assert np.array_equal(trf.state_dict["transform_matrix"], np.eye(3))


@pytest.mark.parametrize("img", [img_6x6_rgb(),])
def test_projection_empty_sampling_no_trf(img):
    dc = slc.DataContainer(img, "I")
    trf = slt.Projection(affine_transforms=slc.Stream(), p=1)
    trf.sample_transform(dc)
    assert np.array_equal(trf.state_dict["transform_matrix"], np.eye(3))


@pytest.mark.parametrize("img", [img_6x6_rgb(),])
def test_projection_empty_sampling_low_prob_trf(img):
    dc = slc.DataContainer(img, "I")
    trf = slt.Projection(affine_transforms=slc.Stream([slt.Rotate(p=0)]), p=1)

    trf.sample_transform(dc)
    assert np.array_equal(trf.state_dict["transform_matrix"], np.eye(3))


@pytest.mark.parametrize("img", [img_6x6_rgb(),])
def test_projection_empty_sampling_from_many_low_prob_trf(img):
    dc = slc.DataContainer(img, "I")
    trf = slt.Projection(
        affine_transforms=slc.Stream(
            [
                slt.Rotate(angle_range=(70, 70), p=0),
                slt.Rotate(angle_range=(10, 10), p=0),
                slt.Rotate(angle_range=(-2, -2), p=0),
            ]
        ),
        p=1,
    )
    trf.sample_transform(dc)
    assert np.array_equal(trf.state_dict["transform_matrix"], np.eye(3))


@pytest.mark.parametrize("img", [img_6x6_rgb(),])
def test_projection_translate_plus_minus_1(img):
    dc = slc.DataContainer(img, "I")
    trf = slt.Projection(
        affine_transforms=slc.Stream(
            [
                slt.Translate(range_x=(1, 1), range_y=(1, 1), p=1),
                slt.Translate(range_x=(-1, -1), range_y=(-1, -1), p=1),
            ]
        ),
        p=1,
    )

    trf.sample_transform(dc)
    assert np.array_equal(trf.state_dict["transform_matrix"], np.eye(3))


def test_gaussian_noise_no_image_throws_value_error():
    trf = slt.Noise(p=1)
    # Setting up the data
    kpts_data = np.array([[0, 0], [0, 5], [1, 3], [2, 0]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, frame=(6, 6))
    dc = slc.DataContainer((kpts,), "P")

    with pytest.raises(ValueError):
        trf(dc)


def test_gaussian_noise_float_gain():
    trf = slt.Noise(gain_range=0.2, p=1)
    assert isinstance(trf.gain_range, tuple)
    assert len(trf.gain_range) == 2
    assert trf.gain_range[0] == 0 and trf.gain_range[1] == 0.2


@pytest.mark.parametrize("img", [img_6x6(),])
def test_salt_and_pepper_no_gain(img):
    trf = slt.SaltAndPepper(gain_range=0.0, p=1)
    dc_res = trf(slc.DataContainer((img.astype(np.uint8),), "I"))
    assert np.array_equal(img, dc_res[0][0])


@pytest.mark.parametrize("pad_size,pad_type", [(2, "z"), ((2, 2), "z"), (2, "r"), ((2, 2), "r"),])
@pytest.mark.parametrize("img, mask", [(img_3x3(), mask_3x3())])
def test_pad_does_not_change_the_data_when_the_image_and_the_mask_are_big(pad_size, pad_type, img, mask):
    dc = slc.DataContainer((img.copy(), mask.copy()), "IM")
    trf = slt.Pad(pad_to=pad_size, padding=pad_type)
    dc_res = trf(dc)

    np.testing.assert_array_equal(dc_res.data[0], img)
    np.testing.assert_array_equal(dc_res.data[1], mask)


@pytest.mark.parametrize("img", [img_3x3(),])
def test_image_doesnt_change_when_gain_0_in_gaussian_noise_addition(img):
    dc = slc.DataContainer((img,), "I")
    trf = slt.Noise(gain_range=(0, 0), p=1)
    dc_res = trf(dc)
    np.testing.assert_array_equal(img, dc_res.data[0])


@pytest.mark.parametrize("scale,expected", [(2, (1, 2)), (2.5, (1, 2.5)), (0.5, (0.5, 1)), (-1, None)])
def test_scale_range_from_number(scale, expected):
    if expected is not None:
        trf = slt.Scale(range_x=scale, range_y=scale)
        assert trf.range_x == expected
        assert trf.range_x == expected
    else:
        with pytest.raises(ValueError):
            slt.Scale(range_x=scale)
        with pytest.raises(ValueError):
            slt.Scale(range_x=None, range_y=scale)


@pytest.mark.parametrize("img", [img_6x6_rgb(),])
@pytest.mark.parametrize(
    "same, scale_x, scale_y, expected",
    [
        (True, (2, 2), (2, 2), (2, 2)),
        (True, (2, 2), (1, 1), (2, 2)),
        (True, (2, 2), None, (2, 2)),
        (True, None, (2, 2), (2, 2)),
        (False, (2, 2), (2, 2), (2, 2)),
        (False, (2, 2), (3, 3), (2, 3)),
        (False, (2, 2), None, (2, 1)),
        (False, None, (2, 2), (1, 2)),
    ],
)
def test_scale_sampling_scale(same, scale_x, scale_y, expected, img):
    dc = slc.DataContainer(img, "I")
    trf = slt.Scale(range_x=scale_x, range_y=scale_y, same=same)
    trf.sample_transform(dc)
    assert expected == (trf.state_dict["scale_x"], trf.state_dict["scale_y"])


@pytest.mark.parametrize(
    "translate,expected", [(2, (-2, 2)), (2.5, (-2.5, 2.5)), (0.5, (-0.5, 0.5)), (-0.5, (-0.5, 0.5)),],
)
def test_translate_range_from_number(translate, expected):
    trf = slt.Translate(range_x=translate, range_y=translate)
    assert trf.range_x == expected
    assert trf.range_y == expected


@pytest.mark.parametrize(
    "trf_cls,trf_params",
    [
        (slt.Noise, {"gain_range": 0.5, "p": 1}),
        (slt.SaltAndPepper, {"p": 1}),
        (slt.GammaCorrection, {"p": 1}),
        (slt.Contrast, {"p": 1}),
        (slt.Brightness, {"p": 1}),
        (slt.Blur, {"p": 1, "blur_type": "g"}),
        (slt.Blur, {"p": 1, "blur_type": "m"}),
        (slt.Blur, {"p": 1, "blur_type": "mo"}),
    ],
)
@pytest.mark.parametrize("img, mask", [(img_3x4(), mask_3x4())])
def test_image_trfs_dont_change_mask_labels_kpts(trf_cls, trf_params, img, mask):
    trf = trf_cls(**trf_params)
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [2, 0]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, frame=(3, 4))
    dc = slc.DataContainer((img, mask, kpts, 1), "IMPL")
    dc_res = trf(dc)

    assert np.all(dc.data[1] == dc_res.data[1])
    assert np.all(dc.data[2].data == dc_res.data[2].data)
    assert dc.data[3] == dc_res.data[3]


@pytest.mark.parametrize("img_0, img_1", [(img_3x4(), img_6x6_rgb())])
def test_brightness_returns_correct_number_of_channels(img_0, img_1):
    trf = slt.Brightness(p=1, brightness_range=(10, 10))
    dc = slc.DataContainer((img_0, img_0, img_1), "III")
    dc_res = trf(dc)

    img1, img2, img3 = dc_res.data

    assert len(img1.shape) == 3
    assert img1.shape[-1] == 1

    assert len(img2.shape) == 3
    assert img2.shape[-1] == 1

    assert len(img3.shape) == 3
    assert img3.shape[-1] == 3


def test_padding_cant_be_float():
    with pytest.raises(TypeError):
        slt.Pad(pad_to=2.5)


def test_reflective_padding_cant_be_applied_to_kpts():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [2, 0]]).reshape((4, 2))
    kpts = slc.Keypoints(kpts_data, frame=(3, 4))
    dc = slc.DataContainer((1, kpts), "LP")
    trf = slt.Pad(pad_to=(10, 10), padding="r")
    with pytest.raises(ValueError):
        trf(dc)


@pytest.mark.parametrize("cutout_crop_size", [(2, 3), (3, 2)])
@pytest.mark.parametrize("img", [img_2x2(),])
def test_crop_or_cutout_size_are_too_big(img, cutout_crop_size):
    dc = slc.DataContainer((img,), "I")
    trf = slt.Crop(crop_to=cutout_crop_size)
    with pytest.raises(ValueError):
        trf(dc)

    trf = slt.CutOut(p=1, cutout_size=cutout_crop_size)
    with pytest.raises(ValueError):
        trf(dc)


@pytest.mark.parametrize("cutout_crop_size", ["123", ("23", 2), (2.5, 2), (2, 2.2)])
def test_wrong_crop_size_types(cutout_crop_size):
    with pytest.raises(TypeError):
        slt.Crop(crop_to=cutout_crop_size)

    with pytest.raises(TypeError):
        slt.CutOut(cutout_size=cutout_crop_size)


@pytest.mark.parametrize("salt_p", [(1, 2), (2, 2), (0.7, 0.3), (-0.1), (-0.8, 0.1), (0.3, 0.7, 0.8)])
def test_wrong_salt_p_salt_and_pepper(salt_p):
    with pytest.raises(ValueError):
        slt.SaltAndPepper(salt_p=salt_p)


@pytest.mark.parametrize(
    "inp, transform_settings, expected",
    [
        (np.ones((5, 5, 1), dtype=np.uint8), None, img_7x7() // 255),
        (np.ones((5, 5, 1), dtype=np.uint8), {0: {"padding": "z"}}, img_7x7() // 255),
        (np.ones((5, 5, 1), dtype=np.uint8), {0: {"padding": "r"}}, np.ones((7, 7, 1))),
    ],
)
def test_padding_for_item(inp, transform_settings, expected):
    dc = slc.DataContainer((inp,), "I", transform_settings)
    dc_res = slt.Pad(pad_to=(7, 7))(dc)
    assert np.array_equal(expected, dc_res.data[0])


@pytest.mark.parametrize(
    "trf, gain_range",
    [
        (slt.SaltAndPepper, (1, 2)),
        (slt.SaltAndPepper, (2, 2)),
        (slt.SaltAndPepper, (0.7, 0.3)),
        (slt.SaltAndPepper, -0.1),
        (slt.SaltAndPepper, (-0.8, 0.1)),
        (slt.SaltAndPepper, (0.3, 0.7, 0.8)),
        (slt.Noise, (1, 2)),
        (slt.Noise, (2, 2)),
        (slt.Noise, (0.7, 0.3)),
        (slt.Noise, -0.1),
        (slt.Noise, (-0.8, 0.1)),
        (slt.Noise, (0.3, 0.7, 0.8)),
    ],
)
def test_wrong_gain_range_in_noises(trf, gain_range):
    with pytest.raises(ValueError):
        trf(gain_range=gain_range)


def test_wrong_types_in_gain_and_salt_p_salt_and_pepper():
    with pytest.raises(TypeError):
        slt.SaltAndPepper(salt_p="2")

    with pytest.raises(TypeError):
        slt.SaltAndPepper(gain_range="2")


@pytest.mark.parametrize("img", [img_6x6_rgb(),])
@pytest.mark.parametrize(
    "scale_x, scale_y, same, expected",
    [
        (None, (2, 2), True, (2, 2)),
        ((2, 2), None, True, (2, 2)),
        (None, None, True, (1, 1)),
        (None, (2, 2), False, (1, 2)),
        ((2, 2), None, False, (2, 1)),
        (None, None, False, (1, 1)),
    ],
)
def test_scale_when_range_x_is_none(scale_x, scale_y, same, expected, img):
    dc = slc.DataContainer(img, "I")
    trf = slt.Scale(range_x=scale_x, range_y=scale_y, same=same, p=1)
    trf.sample_transform(dc)
    assert (trf.state_dict["scale_x"], trf.state_dict["scale_y"]) == expected


@pytest.mark.parametrize("img", [img_6x6_rgb(),])
@pytest.mark.parametrize(
    "translate_x, translate_y, expected", [(None, (2, 2), (0, 2)), ((2, 2), None, (2, 0)), (None, None, (0, 0))],
)
def test_translate_when_range_x_is_none(translate_x, translate_y, expected, img):
    dc = slc.DataContainer(img, "I")
    trf = slt.Translate(range_x=translate_x, range_y=translate_y, p=1)
    trf.sample_transform(dc)
    assert (trf.state_dict["translate_x"], trf.state_dict["translate_y"]) == expected


@pytest.mark.parametrize(
    "param_set",
    [
        {"affine_transforms": "123"},
        {"affine_transforms": 123},
        {"affine_transforms": []},
        {"affine_transforms": slc.Stream([slt.Flip(),])},
        {"affine_transforms": slc.Stream([slt.Flip(),])},
        {"v_range": "123"},
        {"v_range": 123},
        {"v_range": ("123", "456")},
        {"v_range": ((2,), (4,))},
    ],
)
def test_random_projection_raises_type_errors(param_set):
    with pytest.raises(TypeError):
        slt.Projection(**param_set)


@pytest.mark.parametrize(
    "value_range,to_catch",
    [
        ((-1, 1), ValueError),
        ((-1.0, 1.0), ValueError),
        ((1.0, -1.0), ValueError),
        (1000.0, ValueError),
        ((0.1, 0.8, 0.1), ValueError),
        ((0.8, 0.1), ValueError),
        (("123", 0.1), TypeError),
        ((0.1, "123"), TypeError),
    ],
)
def test_lut_transforms_raise_errors(value_range, to_catch):
    with pytest.raises(to_catch):
        slt.GammaCorrection(gamma_range=value_range)

    with pytest.raises(to_catch):
        slt.Contrast(contrast_range=value_range)


@pytest.mark.parametrize(
    "blur_t, k_size, sigma, to_catch",
    [
        (None, "123", None, TypeError),
        (None, (1, 3, 0), None, ValueError),
        (None, (1, 3, 5, -7), None, ValueError),
        (None, (1, 3, 5,), -1, ValueError),
        (None, (1, 3, 5,), "123", TypeError),
        (None, (1, 3, 5,), (0, -4), ValueError),
        (None, (1, 3, 5,), (1, "34"), TypeError),
        ("mo", (1, 1), 1, ValueError),
    ],
)
@pytest.mark.parametrize("img", [img_6x6_rgb(),])
def test_blur_arguments(blur_t, k_size, sigma, to_catch, img):
    dc = slc.DataContainer(img, "I")
    with pytest.raises(to_catch):
        b = slt.Blur(blur_type=blur_t, k_size=k_size, gaussian_sigma=sigma)
        b.sample_transform(dc)


@pytest.mark.parametrize("img", [img_6x6_rgb(),])
@pytest.mark.parametrize(
    "blur_t, k_size, sigma",
    [
        (None, (1, 3), None),
        (None, 3, None),
        (None, (1, 3, 5), None),
        (None, (1, 3, 5,), 1),
        ("g", (1, 3), None),
        ("g", 3, None),
        ("g", (1, 3, 5), None),
        ("g", (1, 3, 5,), 1),
        ("m", (1, 3), None),
        ("m", 3, None),
        ("m", (1, 3, 5), None),
        ("m", (1, 3, 5,), 1),
    ],
)
def test_blur_samples_correctly(blur_t, k_size, sigma, img):
    dc = slc.DataContainer(img, "I")
    trf = slt.Blur(blur_type=blur_t, k_size=k_size, gaussian_sigma=sigma)
    trf.sample_transform(dc)

    if isinstance(k_size, int):
        k_size = (k_size,)
    if sigma is None:
        sigma = 1

    assert trf.state_dict["k_size"] in k_size
    assert trf.state_dict["sigma"] == sigma


@pytest.mark.parametrize("img", [img_6x6(),])
@pytest.mark.parametrize(
    "h_range, s_range, v_range", [((0, 0), None, None), (None, (0, 0), None), (None, None, (0, 0))],
)
def test_hsv_doesnt_change_an_image(h_range, s_range, v_range, img):
    trf = slt.HSV(p=1, h_range=h_range, s_range=s_range, v_range=v_range)
    img_rgb = np.dstack((img, img, img)).astype(np.uint8) * 255
    dc = slc.DataContainer(img_rgb, "I")

    dc_res = trf(dc)

    assert 0 == trf.state_dict["h_mod"]
    assert 0 == trf.state_dict["s_mod"]
    assert 0 == trf.state_dict["v_mod"]

    np.testing.assert_array_equal(img_rgb, dc_res.data[0])


@pytest.mark.parametrize("dtype", [np.float16, np.int32, np.float64, np.int64])
@pytest.mark.parametrize("img", [img_6x6(),])
def test_hsv_trying_use_not_uint8(dtype, img):
    trf = slt.HSV(p=1)
    img_rgb = np.dstack((img, img, img)).astype(dtype)
    dc = slc.DataContainer(img_rgb, "I")

    with pytest.raises(TypeError):
        trf(dc)


@pytest.mark.parametrize("img", [img_6x6(),])
def test_hsv_doesnt_work_for_1_channel(img):
    trf = slt.HSV(p=1)
    dc = slc.DataContainer(img.astype(np.uint8), "I")

    with pytest.raises(ValueError):
        trf(dc)


def test_intensity_remap_values():
    trf = slt.IntensityRemap(p=1)
    img = np.arange(0, 256, 1, dtype=np.uint8).reshape((16, 16, 1))
    dc = slc.DataContainer(img, "I")
    out = trf(dc).data[0]

    # Mapping is applied correctly
    img_expected = trf.state_dict["LUT"].reshape((16, 16, 1))
    np.testing.assert_array_equal(out, img_expected)

    # Mapping has a positive trendline
    assert np.sum(np.diff(out.astype(np.float).ravel())) > 0

    # Higher kernel size yields more monotonic mapping
    trf_noisy = slt.IntensityRemap(p=1, kernel_size=1)
    trf_low_pass = slt.IntensityRemap(p=1, kernel_size=5)
    out_noisy = trf_noisy(dc).data[0].astype(np.float)
    out_low_pass = trf_low_pass(dc).data[0].astype(np.float)
    std_noisy = np.std(np.diff(out_noisy.ravel()))
    std_low_pass = np.std(np.diff(out_low_pass.ravel()))
    assert std_low_pass < std_noisy


@pytest.mark.parametrize("img, expected", [(img_3x3(), does_not_raise()), (img_3x3_rgb(), pytest.raises(ValueError))])
def test_intensity_remap_channels(img, expected):
    trf = slt.IntensityRemap(p=1)
    dc = slc.DataContainer(img, "I")

    with expected:
        trf(dc)


@pytest.mark.parametrize("img", [img_3x3(),])
@pytest.mark.parametrize(
    "dtype, expected",
    [
        (np.int8, pytest.raises(ValueError)),
        (np.uint16, pytest.raises(ValueError)),
        (np.float, pytest.raises(ValueError)),
        (np.bool_, pytest.raises(ValueError)),
    ],
)
def test_intensity_remap_dtypes(img, dtype, expected):
    trf = slt.IntensityRemap(p=1)
    dc = slc.DataContainer(img.astype(dtype), "I")

    with expected:
        trf(dc)


@pytest.mark.parametrize(
    "mode, img, expected",
    [
        ("none", img_6x6(), img_6x6()),
        ("none", img_6x6_rgb(), img_6x6_rgb()),
        ("gs2rgb", img_6x6(), img_6x6_rgb()),
        ("gs2rgb", img_6x6_rgb(), img_6x6_rgb()),
        ("rgb2gs", img_6x6_rgb(), img_6x6()),
        ("rgb2gs", img_6x6(), img_6x6()),
    ],
)
def test_gs2rgb_returns_expected_results(mode, img, expected):
    trf = slt.CvtColor(mode=mode, keep_dim=False)
    dc = slc.DataContainer(img, "I")
    dc_res = trf(dc)
    np.testing.assert_array_equal(expected, dc_res.data[0])


@pytest.mark.parametrize(
    "img, expected", [(img_6x6_rgb(), img_6x6_rgb()), (img_6x6(), img_6x6())],
)
def test_cvtcolor_keeps_dimensions(img, expected):
    trf = slt.CvtColor(mode="rgb2gs")
    dc_res = trf({"image": img})
    np.testing.assert_array_equal(expected, dc_res.data[0])


@pytest.mark.parametrize("mode", ["gs2rgb", "rgb2gs"])
@pytest.mark.parametrize("mask", [mask_3x4(),])
def test_image_color_conversion_raises_error(mode, mask):
    trf = slt.CvtColor(mode=mode)
    dc = slc.DataContainer(mask, "I")
    with pytest.raises(ValueError):
        trf(dc)


@pytest.mark.parametrize("keep_dim", [1, 1.1, "s"])
def test_image_color_conversion_keepdim_type(keep_dim):
    with pytest.raises(TypeError):
        slt.CvtColor(keep_dim=keep_dim)


@pytest.mark.parametrize("img", [img_5x5(),])
def test_random_proj_and_selective_stream(img):
    dc = slc.DataContainer((img,), "I")

    ppl = slt.Projection(
        slc.SelectiveStream(
            [
                slt.Rotate(angle_range=(90, 90), p=0),
                slt.Scale(range_y=(0, 0.1), same=True, p=0),
                slt.Shear(range_y=(-0.1, 0.1), p=0),
            ],
            n=3,
        ),
        v_range=(0, 0),
    )

    dc_res = ppl(dc)

    assert np.array_equal(dc.data, dc_res.data)


@pytest.mark.parametrize("img", [img_5x5(),])
def test_random_contrast_multiplies_the_data(img):
    dc = slc.DataContainer((img,), "I")

    ppl = slt.Contrast(p=1, contrast_range=(2, 2))
    dc_res = ppl(dc)

    assert np.array_equal(dc.data[0] * 2, dc_res.data[0])


@pytest.mark.parametrize("img", [img_6x6(),])
@pytest.mark.parametrize(
    "transform_settings", [None, {0: {"interpolation": "nearest"}}, {0: {"interpolation": "bicubic"}}],
)
def test_different_interpolations_per_item_per_transform(img, transform_settings):
    dc = slc.DataContainer((img,), "I", transform_settings=transform_settings)
    resize_to = (15, 10)
    dc_res = slt.Resize(resize_to=resize_to, interpolation="bilinear")(dc)

    interp = ALLOWED_INTERPOLATIONS["bilinear"]
    if transform_settings is not None:
        interp = ALLOWED_INTERPOLATIONS[transform_settings[0]["interpolation"][0]]
    assert np.array_equal(cv2.resize(img, resize_to[::-1], interpolation=interp).reshape(*resize_to, 1), dc_res.data[0])


@pytest.mark.parametrize(
    "img, expected, cut_size",
    [
        (img_7x7(), np.zeros((7, 7, 1), dtype=np.uint8), 7),
        (img_6x6(), np.zeros((6, 6, 1), dtype=np.uint8), 6),
        (img_6x6_rgb(), np.zeros((6, 6, 3), dtype=np.uint8), 6),
        (img_7x7(), np.zeros((7, 7, 1), dtype=np.uint8), 1.0),
        (img_6x6(), np.zeros((6, 6, 1), dtype=np.uint8), 1.0),
        (img_6x6_rgb(), np.zeros((6, 6, 3), dtype=np.uint8), 1.0),
    ],
)
def test_cutout_blacks_out_image(img, expected, cut_size):
    dc = slc.DataContainer((img,), "I")
    trf = slc.Stream([slt.CutOut(p=1, cutout_size=cut_size)])

    dc_res = trf(dc, return_torch=False)

    assert np.array_equal(expected, dc_res.data[0])


@pytest.mark.parametrize("img", [np.ones((2, 2, 1)),])
@pytest.mark.parametrize("cut_size", [1, (1, 1), 0.5, (0.5, 0.5)])
def test_cutout_1x1_blacks_corner_pixels_2x2_img(img, cut_size):
    dc = slc.DataContainer((img,), "I")
    trf = slc.Stream([slt.CutOut(p=1, cutout_size=cut_size)])
    dc_res = trf(dc, return_torch=False)

    equal = 0
    for i in range(2):
        for j in range(2):
            tmp_opt = img.copy()
            tmp_opt[i, j] = 0
            if np.array_equal(dc_res.data[0], tmp_opt):
                equal += 1

    assert equal == 1


@pytest.mark.parametrize(
    "jitter_x,jitter_y,exp_x,exp_y", [(-0.5, -0.5, 0, 0), (-0.5, 0.5, 0, 1), (0.5, 0.5, 1, 1), (0, 0, 1, 1)],
)
def test_keypoint_jitter_works_correctly(jitter_x, jitter_y, exp_x, exp_y):
    kpts_data = np.array([[1, 1],]).reshape((1, 2))
    kpts = slc.Keypoints(kpts_data.copy(), frame=(2, 2))

    dc = slc.DataContainer((kpts,), "P")
    trf = slc.Stream([slt.KeypointsJitter(p=1, dx_range=(jitter_x, jitter_x), dy_range=(jitter_y, jitter_y))])
    dc_res = trf(dc, return_torch=False)

    assert np.array_equal(dc_res.data[0].data, np.array([exp_x, exp_y]).reshape((1, 2)))


@pytest.mark.parametrize("img, mask", [(img_3x3(), mask_3x3()),])
def test_keypoint_jitter_does_not_change_img_mask_or_target(img, mask):
    trf = slc.Stream([slt.KeypointsJitter(p=1, dx_range=(-0.2, 0.2), dy_range=(-0.2, 0.2))])
    dc_res = trf({"image": img.copy(), "mask": mask.copy(), "label": 1}, return_torch=False)

    assert np.array_equal(dc_res.data[0], img)
    assert np.array_equal(dc_res.data[1], mask)
    assert np.array_equal(dc_res.data[2], 1)


@pytest.mark.parametrize("ks", [3, (3, 3), 5, (5, 5)])
@pytest.mark.parametrize("img", [img_6x6_rgb(),])
def test_motion_blur_samples_transform(ks, img):
    blur = slt.Blur(p=1, blur_type="mo", k_size=ks)
    random.seed(42)
    dc = slc.DataContainer(img, "I")
    for i in range(100):
        blur.sample_transform(dc)
        if isinstance(ks, int):
            assert blur.state_dict["motion_kernel"].shape[0] == ks
        else:
            assert blur.state_dict["motion_kernel"].shape == ks


@pytest.mark.parametrize("img", [img_6x6_rgb(),])
@pytest.mark.parametrize(
    "quality, different", [(None, False), (90, True), (0.9, True), ((50, 90), True), ((0.5, 0.9), True), (50, True)],
)
def test_jpeg_transform(img, quality, different):
    trf = slt.JPEGCompression(quality_range=quality, p=1)
    dc_res = trf({"image": img})
    assert (not np.array_equal(img, dc_res.data[0])) == different


@pytest.mark.parametrize("quality", ["1", (0.4, 1), (10, 20.0)])
def test_jpeg_quality_range_raises_error_when_wrong(quality):
    with pytest.raises(TypeError):
        slt.JPEGCompression(quality_range=quality, p=1)


@pytest.mark.parametrize(
    "img, d_range, rotate, ratio, expected", [(np.ones((3, 3, 3)), (5, 10), 1, 0.2, np.zeros((3, 3, 3))),],
)
def test_gridmask_transform(img, d_range, rotate, ratio, expected):
    np.random.seed(42)
    random.seed(42)
    dc = slc.DataContainer((img,), "I")
    trf = slc.Stream([slt.GridMask(d_range=d_range, rotate=rotate, ratio=ratio, p=1)])

    dc_res = trf(dc, return_torch=False)
    assert np.array_equal(expected, dc_res.data[0])


@pytest.mark.parametrize("d_range", [(-1, 0)])
def test_gridmask_d_range_raises_error_when_wrong(d_range):
    with pytest.raises(ValueError):
        slt.GridMask(d_range=d_range, p=1)


@pytest.mark.parametrize("rotate", [0.1])
def test_gridmask_rotate_raises_error_when_wrong(rotate):
    with pytest.raises(TypeError):
        slt.GridMask(rotate=rotate, p=1)


@pytest.mark.parametrize("rotate", [(0.1, 0.1)])
def test_gridmask_rotate_raises_error_when_wrong_tuple(rotate):
    with pytest.raises(TypeError):
        slt.GridMask(rotate=rotate, p=1)
