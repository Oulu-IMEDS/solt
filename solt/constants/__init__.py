"""
Constants module

.. data:: allowed_blurs

    Defines the blur modes. Can be `'g'` - gaussian or `'m'` - median.

.. data:: allowed_types

    Defines the allowed types to be stored in a DataCantainer. Can be
    `'I'` - image, `'M'` - mask, `'L'` - labels, `'P'` - Keypoints.

.. data:: allowed_crops

    Defines the allowed crops. Can be `'r'` - random crop or `'c'` - center crop.

.. data:: allowed_paddings

    Defines the allowed crops. Can be `'z'` - zero padding or `'r'` - reflective padding.

.. data:: allowed_interpolations

    Defines the allowed interpolation modes. Can be `'bilinear'`, `'nearest'` or `'bicubic'`.

.. data:: dtypes_max

    Defines the maximums for different data types. Can be numpy.dtype('uint8') or numpy.dtype('uint16').

.. data:: allowed_color_conversions

    Defines the allowed color conversion modes. Can be `'gs2rgb'`, `'rgb2gs'` or `'none'`.


"""

from ._constants import allowed_blurs, allowed_types, allowed_crops, \
    allowed_paddings, allowed_interpolations, dtypes_max, allowed_color_conversions

__all__ = ['allowed_blurs', 'allowed_types', 'allowed_crops',
           'allowed_paddings', 'allowed_interpolations',
           'dtypes_max', 'allowed_color_conversions']
