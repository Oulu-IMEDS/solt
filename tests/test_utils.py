import solt.utils as slu
import solt.transforms as slt
import solt.core as slc
import pytest
import json


def test_parameter_validation_range_default_value_not_tuple():
    with pytest.raises(TypeError):
        slu.validate_numeric_range_parameter(123, 123)


def test_parameter_validation_raises_error_when_types_dont_match():
    with pytest.raises(NotImplementedError):
        slu.validate_parameter([1, 2], {1, 2}, 10, int)


def test_parameter_validation_raises_error_when_default_type_is_wrong():
    with pytest.raises(ValueError):
        slu.validate_parameter(None, {1, 2}, (10, '12345'), int)


def test_parameter_validation_raises_error_when_default_value_is_wrong_type():
    with pytest.raises(TypeError):
        slu.validate_parameter(None, {1, 2}, ('10', 'inherit'), int)


@pytest.mark.parametrize('parameter', [
    (1, 2, 3),
    (10, 'inherit'),
    (1, 'i'),
]
                         )
def test_validate_parameter_raises_value_errors(parameter):
    with pytest.raises(ValueError):
        slu.validate_parameter(parameter, {1, 2}, 1, basic_type=int)



@pytest.mark.parametrize('serialized', [
    {'stream':
        {'transforms': [
            {'pad': {
                'pad_to': 34
            }},
            {'crop': {
                'crop_to': 32,
                'crop_mode': 'r'
            }},
            {'cutout': {
                'cutout_size': 2
            }}
        ]}},
    {'stream':
         {'interpolation': None,
          'padding': None,
          'transforms': [
              {'pad': {
                  'pad_to': 34,
                  'padding': 'z'
              }},
              {'crop': {
                  'crop_to': 32,
                  'crop_mode': 'r'
              }},
              {'cutout': {
                  'cutout_size': 2,
                  'p': 0.5
              }}
          ]},
     },

])
def test_deserialize_from_dict(serialized):
    trfs = slc.Stream([
        slt.Pad(34),
        slt.Crop(32, 'r'),
        slt.CutOut(2)
    ])

    serialized_trfs = json.dumps(trfs.to_dict())
    serialized_from_deserialized = json.dumps(slu.from_dict(serialized).to_dict())

    assert serialized_trfs == serialized_from_deserialized


@pytest.mark.parametrize('serialized, stream', [
    [{'stream':
        {'transforms': [
            {'pad': {
                'pad_to': 34
            }},
            {'crop': {
                'crop_to': 32,
                'crop_mode': 'r'
            }},
            {'cutout': {
                'cutout_size': 2
            }},
            {'stream':
                 {'interpolation': None,
                  'padding': None,
                  'transforms': [
                      {'pad': {
                          'pad_to': 34,
                          'padding': 'z'
                      }},
                      {'crop': {
                          'crop_to': 32,
                          'crop_mode': 'r'
                      }},
                      {'cutout': {
                          'cutout_size': 2,
                          'p': 0.5
                      }}
                  ]},
             },
        ]}}, slc.Stream([
        slt.Pad(34),
        slt.Crop(32, 'r'),
        slt.CutOut(2),
        slc.Stream([
            slt.Pad(34),
            slt.Crop(32, 'r'),
            slt.CutOut(2)
        ])
    ])],
    [{'stream':
        {'transforms': [
            {'stream':
                 {'interpolation': None,
                  'padding': None,
                  'transforms': [
                      {'pad': {
                          'pad_to': 34,
                          'padding': 'z'
                      }},
                      {'crop': {
                          'crop_to': 32,
                          'crop_mode': 'r'
                      }},
                      {'cutout': {
                          'cutout_size': 2,
                          'p': 0.5
                      }}
                  ]},
             },
            {'pad': {
                'pad_to': 34
            }},
            {'crop': {
                'crop_to': 32,
                'crop_mode': 'c'
            }},
            {'cutout': {
                'cutout_size': 4
            }},
            {'projection': {
                'v_range': (0, 1e-3),
                'affine_transforms': {
                    'stream': {
                        'transforms': [
                            {'rotate': {'angle_range': 30}},
                            {'scale': {'range_x': 2, 'same': True}}
                        ],
                    }
                }
            }}
        ]}},
        slc.Stream([
            slc.Stream([
                slt.Pad(34),
                slt.Crop(32, 'r'),
                slt.CutOut(2)
            ]),
            slt.Pad(34),
            slt.Crop(32, 'c'),
            slt.CutOut(4),
            slt.Projection(slc.Stream([slt.Rotate(30), slt.Scale(2)]), v_range=(0, 1e-3)),
        ])
    ],
])
def test_deserialize_from_dict_nested(serialized: dict, stream: slc.Stream):
    serialized_trfs = json.dumps(stream.to_dict())
    serialized_from_deserialized = json.dumps(slu.from_dict(serialized).to_dict())

    assert serialized_trfs == serialized_from_deserialized



def test_stream_serializes_all_args_are_set():
    ppl = slc.Stream([
        slt.Rotate(angle_range=(-106, 90), p=0.7, interpolation='nearest'),
        slt.Rotate(angle_range=(-106, 90), p=0.7, interpolation='nearest'),
        slt.Rotate(angle_range=(-106, 90), p=0.7, interpolation='nearest'),
        slt.Projection(
            slc.Stream([
                slt.Rotate(angle_range=(-6, 90), p=0.2, padding='r', interpolation='nearest'),
            ])
        )
    ])

    serialized = ppl.to_dict()
    assert 'interpolation' in serialized
    assert 'padding' in serialized
    assert 'optimize_stack' in serialized
    assert 'transforms' in serialized
    assert len(serialized) == 4

    trfs = serialized['transforms']
    for i, el in enumerate(trfs):
        t = list(el.keys())[0]
        if i < len(serialized) - 1:
            assert list(el.keys())[0] == 'Rotate'
            assert trfs[i][t]['p'] == 0.7
            assert trfs[i][t]['interpolation'] == ('nearest', 'inherit')
            assert trfs[i][t]['padding'] == ('z', 'inherit')
            assert trfs[i][t]['angle_range'] == (-106, 90)
        else:
            assert t == 'Projection'
            assert trfs[i][t]['affine_transforms']['stream']['transforms'][0]['Rotate']['p'] == 0.2
            assert trfs[i][t]['affine_transforms']['stream']['transforms'][0]['Rotate']['interpolation'] == ('nearest', 'inherit')
            assert trfs[i][t]['affine_transforms']['stream']['transforms'][0]['Rotate']['padding'] == ('r', 'inherit')
            assert trfs[i][t]['affine_transforms']['stream']['transforms'][0]['Rotate']['angle_range'] == (-6, 90)
