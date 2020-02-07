from solt.utils import validate_parameter, validate_numeric_range_parameter
import solt.transforms as slt
from solt.core import Stream
import pytest


def test_parameter_validation_range_default_value_not_tuple():
    with pytest.raises(TypeError):
        validate_numeric_range_parameter(123, 123)


def test_parameter_validation_raises_error_when_types_dont_match():
    with pytest.raises(NotImplementedError):
        validate_parameter([1, 2], {1, 2}, 10, int)


def test_parameter_validation_raises_error_when_default_type_is_wrong():
    with pytest.raises(ValueError):
        validate_parameter(None, {1, 2}, (10, '12345'), int)


def test_parameter_validation_raises_error_when_default_value_is_wrong_type():
    with pytest.raises(TypeError):
        validate_parameter(None, {1, 2}, ('10', 'inherit'), int)


@pytest.mark.parametrize('parameter', [
    (1, 2, 3),
    (10, 'inherit'),
    (1, 'i'),
    ]
)
def test_validate_parameter_raises_value_errors(parameter):
    with pytest.raises(ValueError):
        validate_parameter(parameter, {1, 2}, 1, basic_type=int)


@pytest.mark.parametrize('obj', [slt.RandomRotate, Stream, ])
@pytest.mark.parametrize('fmt', ['j', 'jsonn', 'ffmt', 'ftm', 0, 5.2])
def test_incorrect_serialization_format(obj, fmt):
    obj = obj()
    with pytest.raises(ValueError):
        obj.serialize(fmt)
