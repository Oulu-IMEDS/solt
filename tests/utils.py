import numpy as np


def gen_gs_img_black_edge(h, w):
    img = np.ones((h, w, 1))
    img[:, 0] = 0
    img[:, -1] = 0
    img[0, :] = 0
    img[-1, :] = 0
    return img.astype(np.uint8) * 255
