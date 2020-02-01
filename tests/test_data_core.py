import solt.core as slc
import solt.data as sld
import solt.transforms as slt
import solt.utils as slu
import numpy as np
import pytest
import cv2
import random

from .fixtures import img_2x2, img_3x4, mask_2x2, mask_3x4, img_5x5, mask_5x5


def test_img_shape_checker_decorator_shape_check():
    img = np.random.rand(3, 4, 5, 6)
    func = slu.img_shape_checker(lambda x: x)
    with pytest.raises(ValueError):
        func(img)


def test_data_container_different_length_of_data_and_format(img_2x2):
    with pytest.raises(ValueError):
        sld.DataContainer((img_2x2,), 'II')


def test_data_container_create_from_any_data(img_2x2):
    d = sld.DataContainer(img_2x2, 'I')
    assert np.array_equal(img_2x2, d.data[0])
    assert d.data_format == 'I'


def test_data_container_can_be_only_tuple_if_iterable_single(img_2x2):
    with pytest.raises(TypeError):
        sld.DataContainer([img_2x2, ], 'I')


def test_data_container_can_be_only_tuple_if_iterable_multple(img_2x2):
    with pytest.raises(TypeError):
        sld.DataContainer([img_2x2, img_2x2], 'II')


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
    res, _, _ = stream(dc)[0]
    assert np.all(res == img)


def test_empty_stream_selective():
    with pytest.raises(ValueError):
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
    img_res, t0, _ = dc[0]
    mask_res, t1, _ = dc[1]

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
    img_res, _, _ = dc[0]

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
    ], optimize_stack=True)

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
        ], optimize_stack=True)
    ], optimize_stack=True)

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
        slt.RandomRotate(rotation_range=(-90, 90)),
        slt.RandomProjection(
            slc.Stream([
                slt.RandomRotate(rotation_range=(-90, 90)),
            ])
        )
    ])

    serialized = ppl.serialize()

    for i, el in enumerate(serialized):
        if i < len(serialized) - 1:
            assert el == 'RandomRotate'
            assert serialized[el]['p'] == 0.5
            assert serialized[el]['interpolation'] == ('bilinear', 'inherit')
            assert serialized[el]['padding'] == ('z', 'inherit')
            assert serialized[el]['range'] == (-90, 90)
        else:
            assert el == 'RandomProjection'
            assert serialized[el]['transforms']['RandomRotate']['p'] == 0.5
            assert serialized[el]['transforms']['RandomRotate']['interpolation'] == ('bilinear', 'inherit')
            assert serialized[el]['transforms']['RandomRotate']['padding'] == ('z', 'inherit')
            assert serialized[el]['transforms']['RandomRotate']['range'] == (-90, 90)


def test_selective_pipeline_selects_transforms_and_does_the_fusion():
    ppl = slc.SelectiveStream([
        slt.RandomRotate(rotation_range=(90, 90), p=1),
        slt.RandomRotate(rotation_range=(-90, -90), p=1),
    ], n=2, probs=[0.5, 0.5], optimize_stack=True)

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


def test_putting_wrong_format_in_data_container(img_2x2):
    with pytest.raises(TypeError):
        sld.DataContainer(img_2x2, 'Q')


def test_selective_stream_too_many_probs():
    with pytest.raises(ValueError):
        slc.SelectiveStream([
            slt.RandomRotate(rotation_range=(90, 90), p=1),
            slt.RandomRotate(rotation_range=(-90, -90), p=1),
        ], n=2, probs=[0.4, 0.3, 0.3])


def test_selective_stream_low_prob_transform_should_not_change_the_data(img_5x5):
    img = img_5x5
    dc = sld.DataContainer((img,), 'I')

    ppl = slc.SelectiveStream([
        slt.RandomRotate(rotation_range=(90, 90), p=0),
        slt.RandomRotate(rotation_range=(-90, -90), p=0)
    ])

    dc_res = ppl(dc)

    np.array_equal(dc.data, dc_res.data)


def test_manually_specified_padding_and_interpolation(img_5x5, mask_5x5):
    dc = sld.DataContainer((img_5x5, img_5x5, mask_5x5, mask_5x5, 1), 'IIMML',
                           {0: {'interpolation': 'bicubic', 'padding': 'z'},
                            2: {'interpolation': 'bilinear'},
                            3: {'padding': 'r'}
                            })

    assert dc.transform_settings[0]['interpolation'] == ('bicubic', 'strict')
    assert dc.transform_settings[1]['interpolation'] == ('bilinear', 'inherit')
    assert dc.transform_settings[2]['interpolation'] == ('bilinear', 'strict')
    assert dc.transform_settings[3]['interpolation'] == ('nearest', 'strict')

    assert dc.transform_settings[0]['padding'] == ('z', 'strict')
    assert dc.transform_settings[1]['padding'] == ('z', 'inherit')
    assert dc.transform_settings[2]['padding'] == ('z', 'inherit')
    assert dc.transform_settings[3]['padding'] == ('r', 'strict')


def test_transform_settings_wrong_type(img_5x5):
    with pytest.raises(TypeError):
        sld.DataContainer((img_5x5, img_5x5, 1), 'IIL', ())


def test_transform_settings_wrong_length(img_5x5):
    with pytest.raises(ValueError):
        sld.DataContainer((img_5x5, img_5x5, 1), 'IIL', {1: {}, 2: {}, 3: {}, 4: {}})


def test_transform_settings_wrong_type_for_item(img_5x5):
    with pytest.raises(TypeError):
        sld.DataContainer((img_5x5, img_5x5, 1), 'IIL', {1: 123, 0: None})


@pytest.mark.parametrize('setting', [
    {'interpolation': 'bilinear'},
    {'interpolation': 'bilinear', 'padding': 'z'},
    {'padding': 'z'}
])
def test_interpolation_or_padding_settings_for_labels_or_keypoints(setting):
    kpts = sld.KeyPoints(pts=np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2)), H=3, W=3)
    with pytest.raises(TypeError):
        sld.DataContainer(data=(kpts,),
                          fmt='P',
                          transform_settings={0: setting})


@pytest.mark.parametrize('ignore_state', [True, False])
@pytest.mark.parametrize('pipeline', [True, False])
def test_matrix_transforms_state_reset(img_5x5, ignore_state, pipeline):
    n_iter = 50
    if pipeline:
        ppl = slc.Stream([
            slt.RandomRotate(rotation_range=(-180, 180), p=1, ignore_state=ignore_state),
            slt.PadTransform(pad_to=(10, 10)),
        ])
    else:
        ppl = slt.RandomRotate(rotation_range=(-180, 180), p=1, ignore_state=ignore_state)

    img_test = img_5x5.copy()
    img_test[0, 0] = 1
    random.seed(42)

    trf_not_eq = 0
    imgs_not_eq = 0
    for i in range(n_iter):
        dc1 = sld.DataContainer((img_test.copy(),), 'I')
        dc2 = sld.DataContainer((img_test.copy(),), 'I')

        dc1_res = ppl(dc1).data[0].squeeze()
        if pipeline:
            trf_state1 = ppl.transforms[0].state_dict['transform_matrix_corrected']
        else:
            trf_state1 = ppl.state_dict['transform_matrix_corrected']

        dc2_res = ppl(dc2).data[0].squeeze()
        if pipeline:
            trf_state2 = ppl.transforms[0].state_dict['transform_matrix_corrected']
        else:
            trf_state2 = ppl.state_dict['transform_matrix_corrected']

        if not np.array_equal(trf_state1, trf_state2):
            trf_not_eq += 1

        if not np.array_equal(dc1_res, dc2_res):
            imgs_not_eq += 1

    random.seed(None)
    assert trf_not_eq > n_iter//2
    assert imgs_not_eq > n_iter//2


@pytest.mark.parametrize('pipeline', [True, False])
def test_matrix_transforms_use_cache_for_different_dc_items_raises_error(img_5x5, mask_3x4, pipeline):
    dc = sld.DataContainer((img_5x5, mask_3x4), 'IM')
    if pipeline:
        ppl = slc.Stream([
            slt.RandomRotate(rotation_range=(-180, 180), p=1, ignore_state=False),
            slt.PadTransform(pad_to=(10, 10)),
        ])
    else:
        ppl = slt.RandomRotate(rotation_range=(-180, 180), p=1, ignore_state=False)

    with pytest.raises(ValueError):
        ppl(dc)


def test_keypoints_get_set():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = sld.KeyPoints(kpts_data, 3, 4)

    assert np.array_equal(kpts[0], np.array([0, 0]))
    kpts[0] = np.array([2, 2])
    assert np.array_equal(kpts[0], np.array([2, 2]))

    with pytest.raises(TypeError):
        kpts[0] = [2, 2]
