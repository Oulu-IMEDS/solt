import solt.core as slc
import solt.data as sld
import solt.transforms as slt
import numpy as np
import pytest
import cv2

from .fixtures import img_2x2, img_3x4, mask_2x2, mask_3x4, img_5x5


def test_data_container_different_length_of_data_and_format(img_2x2):
    with pytest.raises(AssertionError):
        sld.DataContainer((img_2x2,), 'II')


def test_data_container_create_from_any_data(img_2x2):
    d = sld.DataContainer(img_2x2, 'I')
    assert np.array_equal(img_2x2, d.data[0])


def test_data_item_create_img(img_2x2):
    img = img_2x2
    dc = sld.DataContainer((img,), 'I')
    assert len(dc) == 1
    assert np.array_equal(img, dc[0][0])
    assert dc[0][1] == 'I'


def test_stream_empty(img_2x2):
    img = img_2x2
    dc = sld.DataContainer((img,), 'I')
    stream = slc.Stream()
    res, _ = stream(dc)[0]
    assert np.all(res == img)


def test_empty_stream_selective():
    with pytest.raises(AssertionError):
        slc.SelectiveStream()


def test_nested_stream(img_3x4, mask_3x4):
    img, mask = img_3x4, mask_3x4
    dc = sld.DataContainer((img, mask), 'IM')

    stream = slc.Stream([
        slt.RandomFlip(p=1, axis=0),
        slt.RandomFlip(p=1, axis=1),
        slc.Stream([
            slt.RandomFlip(p=1, axis=1),
            slt.RandomFlip(p=1, axis=0),
        ])
    ])

    dc = stream(dc)
    img_res, t0 = dc[0]
    mask_res, t1 = dc[1]

    assert np.array_equal(img, img_res)
    assert np.array_equal(mask, mask_res)


def test_image_shape_equal_3_after_nested_flip(img_3x4):
    img = img_3x4
    dc = sld.DataContainer((img,), 'I')

    stream = slc.Stream([
        slt.RandomFlip(p=1, axis=0),
        slt.RandomFlip(p=1, axis=1),
        slc.Stream([
            slt.RandomFlip(p=1, axis=1),
            slt.RandomFlip(p=1, axis=0),
        ])
    ])

    dc = stream(dc)
    img_res, _ = dc[0]

    assert np.array_equal(len(img.shape), 3)


def test_create_empty_keypoints():
    kpts = sld.KeyPoints()
    assert kpts.H is None
    assert kpts.W is None
    assert kpts.data is None


def test_create_4_keypoints():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = sld.KeyPoints(kpts_data, 3, 4)
    assert kpts.H == 3
    assert kpts.W == 4
    assert np.array_equal(kpts_data, kpts.data)


def test_create_4_keypoints_change_frame():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = sld.KeyPoints(kpts_data, 3, 4)
    kpts.H = 2
    kpts.W = 2

    assert kpts.H == 2
    assert kpts.W == 2
    assert np.array_equal(kpts_data, kpts.data)


def test_create_4_keypoints_change_grid_and_frame():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = sld.KeyPoints(kpts_data, 3, 4)

    kpts_data_new = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5]]).reshape((5, 2))
    kpts.H = 2
    kpts.W = 2
    kpts.pts = kpts_data_new

    assert kpts.H == 2
    assert kpts.W == 2
    assert np.array_equal(kpts_data_new, kpts.pts)


def test_fusion_happens():
    ppl = slc.Stream([
        slt.RandomScale((0.5, 1.5), (0.5, 1.5), p=1),
        slt.RandomRotate((-50, 50), padding='z', p=1),
        slt.RandomShear((-0.5, 0.5), (-0.5, 0.5), padding='z', p=1),
        slt.RandomFlip(p=1, axis=1),
    ])

    st = ppl.optimize_stack(ppl.transforms)
    assert len(st) == 2


def test_fusion_rotate_360(img_5x5):
    img = img_5x5
    dc = sld.DataContainer((img,), 'I')

    ppl = slc.Stream([
        slt.RandomRotate((45, 45), padding='z', p=1),
        slt.RandomRotate((45, 45), padding='z', p=1),
        slt.RandomRotate((45, 45), padding='z', p=1),
        slt.RandomRotate((45, 45), padding='z', p=1),
        slt.RandomRotate((45, 45), padding='z', p=1),
        slt.RandomRotate((45, 45), padding='z', p=1),
        slt.RandomRotate((45, 45), padding='z', p=1),
        slt.RandomRotate((45, 45), padding='z', p=1),
    ])

    img_res = ppl(dc)[0][0]

    np.testing.assert_array_almost_equal(img, img_res)


def test_fusion_rotate_360_flip_rotate_360(img_5x5):
    img = img_5x5
    dc = sld.DataContainer((img,), 'I')

    ppl = slc.Stream([
        slt.RandomRotate((45, 45), padding='z', p=1),
        slt.RandomRotate((45, 45), padding='z', p=1),
        slt.RandomRotate((45, 45), padding='z', p=1),
        slt.RandomRotate((45, 45), padding='z', p=1),
        slt.RandomRotate((45, 45), padding='z', p=1),
        slt.RandomRotate((45, 45), padding='z', p=1),
        slt.RandomRotate((45, 45), padding='z', p=1),
        slt.RandomRotate((45, 45), padding='z', p=1),
        slt.RandomFlip(p=1, axis=1),
        slc.Stream([
            slt.RandomRotate((45, 45), padding='z', p=1),
            slt.RandomRotate((45, 45), padding='z', p=1),
            slt.RandomRotate((45, 45), padding='z', p=1),
            slt.RandomRotate((45, 45), padding='z', p=1),
            slt.RandomRotate((45, 45), padding='z', p=1),
            slt.RandomRotate((45, 45), padding='z', p=1),
            slt.RandomRotate((45, 45), padding='z', p=1),
            slt.RandomRotate((45, 45), padding='z', p=1),
        ])
    ])

    img_res = ppl(dc)[0][0]

    np.testing.assert_array_almost_equal(cv2.flip(img, 1).reshape(5, 5, 1), img_res)


def test_stream_settings():
    ppl = slc.Stream([
        slt.RandomRotate((45, 45), interpolation='bicubic', padding='z', p=1),
        slt.RandomRotate((45, 45), padding='r', p=1),
        slt.RandomRotate((45, 45), interpolation='bicubic', padding='z', p=1),
        slt.RandomShear(0.1, 0.1, interpolation='bilinear', padding='z'),
        ],
        interpolation='nearest',
        padding='z'
    )

    for trf in ppl.transforms:
        assert trf.interpolation[0] == 'nearest'
        assert trf.padding[0] == 'z'


def test_stream_settings_replacement():
    ppl = slc.Stream([
        slt.RandomRotate((45, 45), interpolation='bicubic', padding='z', p=1),
        slt.RandomRotate((45, 45), padding='r', p=1),
        slt.RandomRotate((45, 45), interpolation='bicubic', padding='z', p=1),
        slt.RandomShear(0.1, 0.1, interpolation='bilinear', padding='z'),
    ],
        interpolation='nearest',
        padding='z'
    )

    ppl.interpolation = 'bilinear'
    ppl.padding = 'r'

    for trf in ppl.transforms:
        assert trf.interpolation[0] == 'bilinear'
        assert trf.padding[0] == 'r'


def test_stream_settings_strict():
    ppl = slc.Stream([
        slt.RandomRotate((45, 45), interpolation='bicubic', padding='z', p=1),
        slt.RandomRotate((45, 45), padding='r', p=1),
        slt.RandomRotate((45, 45), interpolation=('bicubic', 'strict'), padding=('r', 'strict'), p=1),
        slt.RandomShear(0.1, 0.1, interpolation='bilinear', padding='z'),
        ],
        interpolation='nearest',
        padding='z'
    )

    for idx, trf in enumerate(ppl.transforms):
        if idx == 2:
            assert trf.interpolation[0] == 'bicubic'
            assert trf.padding[0] == 'r'
        else:
            assert trf.interpolation[0] == 'nearest'
            assert trf.padding[0] == 'z'


def test_stream_nested_settings():
    ppl = slc.Stream([
        slt.RandomRotate((45, 45), interpolation='bicubic', padding='z', p=1),
        slt.RandomRotate((45, 45), padding='r', p=1),
        slc.Stream([
            slt.RandomRotate((45, 45), interpolation='bicubic', padding='z', p=1),
            slt.RandomRotate((45, 45), padding='r', p=1),
        ], interpolation='bicubic', padding='r'
        )
        ],
        interpolation='nearest',
        padding='z'
    )

    trfs = ppl.transforms[:2] + ppl.transforms[-1].transforms

    for idx, trf in enumerate(trfs):
        assert trf.interpolation[0] == 'nearest'
        assert trf.padding[0] == 'z'


def test_stream_raises_assertion_error_when_not_basetransform_or_stream_in_the_transforms():
    with pytest.raises(TypeError):
        slc.Stream([1,2,3])


def test_stream_serializes_correctly():
    ppl = slc.Stream([
        slt.RandomRotate(rotation_range=(-90,90)),
        slt.RandomRotate(rotation_range=(-90, 90)),
        slt.RandomRotate(rotation_range=(-90, 90))
    ])

    serialized = ppl.serialize()

    for el in serialized:
        assert el == 'RandomRotate'
        assert serialized['RandomRotate']['p'] == 0.5
        assert serialized['RandomRotate']['interpolation'] == ('bilinear', 'inherit')
        assert serialized['RandomRotate']['padding'] == ('z', 'inherit')
        assert serialized['RandomRotate']['range'] == (-90, 90)


def test_selective_pipeline_selects_transforms_and_does_the_fusion():
    ppl = slc.SelectiveStream([
        slt.RandomRotate(rotation_range=(90, 90), p=1),
        slt.RandomRotate(rotation_range=(-90, -90), p=1),
    ], n=2, probs=[0.5, 0.5])

    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = sld.KeyPoints(kpts_data, 3, 4)
    dc = sld.DataContainer(kpts, 'P')
    dc_res = ppl(dc)

    assert np.array_equal(np.eye(3), ppl.transforms[0].state_dict['transform_matrix'])


def test_value_error_when_optimizeing_wrong_elements_transforms_list():
    trfs = [
        slt.RandomRotate(rotation_range=(90, 90), p=1),
        slt.RandomRotate(rotation_range=(-90, -90), p=1),
        lambda x: x**2
    ]

    with pytest.raises(TypeError):
        slc.Stream.optimize_stack(trfs)


def test_nested_streams_are_not_fused_with_matrix_trf():
    trfs = [
        slt.RandomRotate(rotation_range=(90, 90), p=1),
        slt.RandomRotate(rotation_range=(-90, -90), p=1),
        slc.Stream([
            slt.RandomRotate(rotation_range=(90, 90), p=1),
        ]),
        slt.RandomRotate(rotation_range=(-90, -90), p=1),
    ]

    trfs_optimized = slc.Stream.optimize_stack(trfs)
    assert trfs_optimized[-2] == trfs[-2]

