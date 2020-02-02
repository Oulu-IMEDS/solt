import solt.core as slc
import solt.data as sld
import solt.transforms as slt
import solt.utils as slu
import numpy as np
import pytest
import cv2
import random
import itertools
import torch

from .fixtures import img_2x2, img_3x4, mask_2x2, mask_3x4, img_5x5, mask_5x5, img_3x3_rgb, mask_3x3


def assert_data_containers_equal(dc, dc_new):
    assert dc_new.data_format == dc.data_format
    for d1, d2 in zip(dc_new.data, dc.data):
        if isinstance(d1, np.ndarray):
            np.testing.assert_array_equal(d1, d2)
        elif isinstance(d1, sld.KeyPoints):
            assert d1.height == d2.height and d1.width == d2.width
            np.testing.assert_array_equal(d1.data, d2.data)
        else:
            assert d1 == d2


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
    assert kpts.height is None
    assert kpts.width is None
    assert kpts.data is None


def test_create_4_keypoints():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = sld.KeyPoints(kpts_data, 3, 4)
    assert kpts.height == 3
    assert kpts.width == 4
    assert np.array_equal(kpts_data, kpts.data)


def test_create_4_keypoints_change_frame():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = sld.KeyPoints(kpts_data, 3, 4)
    kpts.height = 2
    kpts.width = 2

    assert kpts.height == 2
    assert kpts.width == 2
    assert np.array_equal(kpts_data, kpts.data)


def test_create_4_keypoints_change_grid_and_frame():
    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2))
    kpts = sld.KeyPoints(kpts_data, 3, 4)

    kpts_data_new = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5]]).reshape((5, 2))
    kpts.height = 2
    kpts.width = 2
    kpts.pts = kpts_data_new

    assert kpts.height == 2
    assert kpts.width == 2
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
        slc.Stream([1, 2, 3])


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
    kpts = sld.KeyPoints(pts=np.array([[0, 0], [0, 2], [2, 2], [2, 0]]).reshape((4, 2)), height=3, width=3)
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


@pytest.mark.parametrize('order', list(itertools.permutations(['image', 'images', 'mask', 'masks', 'keypoints', 'keypoints_array', 'label', 'labels']))[:20])
@pytest.mark.parametrize('presence', [[1, 2, 1, 2, 1, 0, 1, 2],
                                      [1, 0, 1, 2, 0, 2, 0, 3],
                                      [0, 2, 0, 0, 2, 0, 0, 0],
                                      [0, 2, 0, 2, 0, 2, 0, 2],
                                      [0, 0, 1, 0, 1, 0, 1, 0]])
def test_data_container_from_and_to_dict(img_3x4, mask_3x4, order, presence):
    img, mask = img_3x4, mask_3x4

    kpts_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2)).astype(float)
    kpts = sld.KeyPoints(kpts_data.copy(), 3, 4)

    n_obj1, n_obj2, n_obj3, n_obj4, n_obj5, n_obj6, n_obj7, n_obj8 = presence
    order = list(order)
    d = {}
    dc_content = []
    dc_format = ''
    if n_obj1:
        d['image'] = img
        dc_content.append(img)
        dc_format += 'I'
    else:
        del order[order.index('image')]
    if n_obj2:
        d['images'] = [img.copy() for _ in range(n_obj2)]
        dc_content.extend(d['images'])
        dc_format += 'I' * n_obj2
    else:
        del order[order.index('images')]
    if n_obj3:
        d['mask'] = mask
        dc_content.append(mask)
        dc_format += 'M'
    else:
        del order[order.index('mask')]
    if n_obj4:
        d['masks'] = [mask.copy() for i in range(n_obj4)]
        dc_content.extend(d['masks'])
        dc_format += 'M' * n_obj4
    else:
        del order[order.index('masks')]
    if n_obj5:
        d['keypoints'] = sld.KeyPoints(kpts_data.copy(), 3, 4)
        dc_content.append(kpts)
        dc_format += 'P'
    else:
        del order[order.index('keypoints')]
    if n_obj6:
        d['keypoints_array'] = [sld.KeyPoints(kpts_data.copy(), 3, 4) for _ in range(n_obj6)]
        dc_content.extend(d['keypoints_array'])
        dc_format += 'P' * n_obj6
    else:
        del order[order.index('keypoints_array')]
    if n_obj7:
        d['label'] = 1
        dc_content.append(1)
        dc_format += 'L'
    else:
        del order[order.index('label')]
    if n_obj8:
        d['labels'] = [1 for _ in range(n_obj8)]
        dc_content.extend([1 for _ in range(n_obj8)])
        dc_format += 'L' * n_obj8
    else:
        del order[order.index('labels')]
    dc = sld.DataContainer(tuple(dc_content), dc_format)
    reordered_d = {k: d[k] for k in order}

    # This tests whether the creation from dict works as expected
    dc_new = sld.DataContainer.from_dict(reordered_d)
    assert_data_containers_equal(dc, dc_new)

    # Now we will also test whether conversion to dict and back works well.
    tensor_dict = dc_new.to_torch(as_dict=True, normalize=False, scale_keypoints=False)
    for k in tensor_dict:
        if isinstance(tensor_dict[k], (list, tuple)):
            tmp = []
            for el in tensor_dict[k]:
                tmp.append(el.numpy() if isinstance(el, torch.Tensor) else el)
                if 'imag' in k:
                    tmp[-1] = (tmp[-1].transpose((1, 2, 0)) * 255).astype(np.uint8)
                if 'mask' in k:
                    tmp[-1] = tmp[-1].astype(np.uint8).squeeze()
                if 'keypoints' in k:
                    tmp[-1] = sld.KeyPoints(tmp[-1], 3, 4)

            tensor_dict[k] = tmp
        else:
            el = tensor_dict[k]
            tensor_dict[k] = (el.numpy()).astype(np.uint8) if isinstance(el, torch.Tensor) else el
            if 'imag' in k:
                tensor_dict[k] = (tensor_dict[k].transpose((1, 2, 0)) * 255).astype(np.uint8)
            if 'mask' in k:
                tensor_dict[k] = tensor_dict[k].astype(np.uint8).squeeze()
            if 'keypoints' in k:
                tensor_dict[k] = sld.KeyPoints(tensor_dict[k], 3, 4)

    dc_from_tensor = sld.DataContainer.from_dict(tensor_dict)
    assert_data_containers_equal(dc, dc_from_tensor)


def test_image_mask_pipeline_to_torch(img_3x4, mask_3x4):
    ppl = slc.Stream(
            [
                slt.RandomRotate(rotation_range=(90, 90), p=1),
                slt.RandomRotate(rotation_range=(90, 90), p=1),
            ],
        )
    img, mask = ppl({'image': img_3x4, 'mask': mask_3x4}).to_torch()
    assert img.max().item() == 1
    assert mask.max().item() == 1
    assert isinstance(img, torch.FloatTensor)
    assert isinstance(mask, torch.FloatTensor)


def test_image_mask_pipeline_to_torch_uint16(img_3x4, mask_3x4):
    ppl = slc.Stream(
            [
                slt.RandomRotate(rotation_range=(90, 90), p=1),
                slt.RandomRotate(rotation_range=(90, 90), p=1),
            ],
        )
    img, mask = ppl({'image': (img_3x4 // 255).astype(np.uint16)*65535, 'mask': mask_3x4}).to_torch()
    assert img.max() == 1
    assert mask.max() == 1
    assert isinstance(img, torch.FloatTensor)
    assert isinstance(mask, torch.FloatTensor)


@pytest.mark.parametrize('mean,std', [[None, None], [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)],
                                      [np.array((0.5, 0.5, 0.5)), (0.5, 0.5, 0.5)],
                                      [(0.5, 0.5, 0.5), np.array((0.5, 0.5, 0.5))],
                                      [np.array((0.5, 0.5, 0.5)), np.array((0.5, 0.5, 0.5))]])
def test_image_mask_pipeline_to_torch_normalization(img_3x3_rgb, mask_3x3, mean, std):
    ppl = slc.Stream(
        [
            slt.RandomRotate(rotation_range=(90, 90), p=1),
            slt.RandomRotate(rotation_range=(90, 90), p=1),
        ],
    )
    dc_res = ppl({'image': img_3x3_rgb, 'mask': mask_3x3})
    img, mask = dc_res.to_torch(normalize=True, mean=mean, std=std)

    if mean is None:
        np.testing.assert_almost_equal(img[0, :, :].max().item(), 0.515/0.229)
    else:
        assert img.max() == 1
    assert mask.max() == 1
    assert isinstance(img, torch.FloatTensor)
    assert isinstance(mask, torch.FloatTensor)


@pytest.mark.parametrize('mean,std, expected', [[(0.5, 0.5), (0.5, 0.5, 0.5), ValueError],
                                                [(0.5, 0.5, 0.5), (0.5, 0.5), ValueError],
                                                [(0.5, 0.5, 0.5), '123', TypeError],
                                                ['123', (0.5, 0.5, 0.5), TypeError],
                                                [torch.tensor((0.5, 0.5, 0.5)).byte(), (0.5, 0.5, 0.5), TypeError],
                                                [torch.tensor((0.5, 0.5, 0.5)).byte(),
                                                 torch.tensor((0.5, 0.5, 0.5)).double(), TypeError],
                                                [torch.tensor((0.5, 0.5, 0.5)),
                                                 torch.tensor((0.5, 0.5, 0.5)).double(), TypeError],
                                                [torch.tensor((0.5, 0.5)),
                                                 torch.tensor((0.5, 0.5, 0.5)), ValueError],
                                                [torch.tensor((0.5, 0.5, 0.5)),
                                                 torch.tensor((0.5, 0.5)), ValueError]
                                                ])
def test_image_mask_pipeline_to_torch_checks_mean_type_and_shape_rgb(img_3x3_rgb, mask_3x3, mean, std, expected):
    ppl = slc.Stream(
        [
            slt.RandomRotate(rotation_range=(90, 90), p=1),
            slt.RandomRotate(rotation_range=(90, 90), p=1),
        ],
    )
    dc_res = ppl({'image': img_3x3_rgb, 'mask': mask_3x3})
    with pytest.raises(expected):
        dc_res.to_torch(normalize=True, mean=mean, std=std)


def test_data_container_keypoints_rescale_to_torch():
    kpts_data = np.array([[100, 20], [1023, 80], [20, 20], [100, 700]]).reshape((4, 2))
    kpts = sld.KeyPoints(kpts_data, 768, 1024)
    ppl = slc.Stream()
    dc_res = ppl({'keypoints': kpts, 'label': 1})
    k, label = dc_res.to_torch(normalize=True, scale_keypoints=True)
    assert isinstance(k, torch.FloatTensor)
    np.testing.assert_almost_equal(k.max().item() * 1023, 1023)
    np.testing.assert_almost_equal(k.min().item() * 1023, 20)
    assert label == 1
