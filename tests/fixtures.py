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
def img_mask_2x2():
    """
    Generates 2x2 mask (doesn't have the 3rd dimension compare to an image).

    Returns
    -------
    out : ndarray
        2x2 mask, uint8
    """
    return img_2x2(), np.array([[1, 0], [0, 1]]).reshape((2, 2)).astype(np.uint8)  # Generating the mask as well


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
def img_mask_3x4():
    """
    Generates a mask  3x4

    Returns
    -------
    out : ndarray
        3x4 uint8 image
    """
    img = img_3x4()
    mask = np.array([[0, 1, 1, 1],
                    [0, 1, 1, 0],
                    [0, 1, 1, 0]]).reshape((3, 4)).astype(np.uint8)
    return img, mask


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
def img_mask_3x3():
    """
    Generates a image+mask  3x4

    Returns
    -------
    out : ndarray
        3x4 uint8 image
    """
    img = img_3x3()
    mask = np.array([[1, 1, 1],
                    [1, 1, 1],
                    [0, 1, 1]]).reshape((3, 3)).astype(np.uint8)
    return img, mask


@pytest.fixture
def img_5x5():
    """
    Generates a gs image 5x5. It is all ones, besides the edges

    Returns
    -------
    out : ndarray
        3x4 uint8 image
    """
    img = np.ones((5,5,1))

    img[:, 0] = 2
    img[:, -1] = 2
    img[0, :] = 2
    img[-1, :] = 2
    return img


@pytest.fixture
def img_6x6():
    """
    Generates a gs image 5x5. It is all ones, besides the edges

    Returns
    -------
    out : ndarray
        3x4 uint8 image
    """
    img = np.ones((6, 6, 1))
    img[:, 0] = 2
    img[:, -1] = 2
    img[0, :] = 2
    img[-1, :] = 2
    return img