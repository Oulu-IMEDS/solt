import pytest
import numpy as np


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
def mask_2x2():
    """
    Generates 2x2 mask (doesn't have the 3rd dimension compare to an image).

    Returns
    -------
    out : ndarray
        2x2 mask, uint8
    """
    return np.array([[1, 0], [0, 1]]).reshape((2, 2)).astype(np.uint8)


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
                    [1, 1, 1, 1]]).reshape((3, 4, 1)).astype(np.uint8)
    return img


@pytest.fixture
def mask_3x4():
    """
    Generates a mask  3x4

    Returns
    -------
    out : ndarray
        3x4 uint8 image
    """

    mask = np.array([[0, 1, 1, 1],
                    [0, 1, 1, 0],
                    [0, 1, 1, 0]]).reshape((3, 4)).astype(np.uint8)
    return mask


@pytest.fixture
def img_3x3():
    """
    Generates a grayscale image 3x4

    Returns
    -------
    out : ndarray
        3x4x1 uint8 image
    """
    img = np.array([[0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]]).reshape((3, 3, 1)).astype(np.uint8)
    return img


@pytest.fixture
def mask_3x3():
    """
    Generates a image+mask  3x4

    Returns
    -------
    out : ndarray
        3x4 uint8 image
    """

    mask = np.array([[1, 1, 1],
                    [1, 1, 1],
                    [0, 1, 1]]).reshape((3, 3)).astype(np.uint8)
    return mask


@pytest.fixture
def img_5x5():
    """
    Generates a gs image 5x5. It is all ones, besides the edges

    Returns
    -------
    out : ndarray
        5x5 uint8 image
    """
    img = np.ones((5, 5, 1))

    img[:, 0] = 0
    img[:, -1] = 0
    img[0, :] = 0
    img[-1, :] = 0
    return img.astype(np.uint8)


@pytest.fixture
def mask_5x5():
    """
    Generates a mask 5x5. It is all ones, besides the edges

    Returns
    -------
    out : ndarray
        5x5 uint8 image
    """
    img = np.ones((5, 5, 1))

    img[:, :2] = 2
    img[:, -2:] = 2
    img[:2, :] = 2
    img[-2, :] = 2
    return img.astype(np.uint8)


@pytest.fixture
def img_6x6():
    """
    Generates a gs image 5x5. It is all ones, besides the edges

    Returns
    -------
    out : ndarray
        6x6 uint8 image
    """
    img = np.ones((6, 6, 1))
    img[:, 0] = 0
    img[:, -1] = 0
    img[0, :] = 0
    img[-1, :] = 0
    return img.astype(np.uint8)*255


@pytest.fixture
def img_7x7():
    """
    Generates a gs image 7x7. It is all ones, besides the edges

    Returns
    -------
    out : ndarray
        6x6 uint8 image
    """
    img = np.ones((7, 7, 1))
    img[:, 0] = 0
    img[:, -1] = 0
    img[0, :] = 0
    img[-1, :] = 0
    return img.astype(np.uint8)*255



@pytest.fixture
def mask_6x6():
    """
    Generates a mask 6x6. It is all ones, besides the edges

    Returns
    -------
    out : ndarray
        3x5 uint8 image
    """
    img = np.ones((6, 6))

    img[:, 0] = 0
    img[:, -1] = 0
    img[0, :] = 0
    img[-1, :] = 0
    return img.astype(np.uint8)


@pytest.fixture
def img_6x6_rgb():
    """
    Generates an RGB image 6x6. It is all ones, besides the edges

    Returns
    -------
    out : ndarray
        6x6 uint8 image
    """
    img = np.ones((6, 6, 1))
    img[:, 0] = 0
    img[:, -1] = 0
    img[0, :] = 0
    img[-1, :] = 0
    return np.dstack((img, img, img)).astype(np.uint8)*255

