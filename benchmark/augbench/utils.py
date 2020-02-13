from PIL import Image
import cv2
import argparse
import os
from augbench.constants import DEFAULT_BENCHMARKING_LIBRARIES

cv2.setNumThreads(0)  # noqa E402
cv2.ocl.setUseOpenCL(False)  # noqa E402


def read_img_pillow(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def read_img_cv2(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def parse_args():
    parser = argparse.ArgumentParser(description="Augmentation libraries performance augbench")
    parser.add_argument(
        "-d", "--data-dir", metavar="DIR", default=os.environ.get("DATA_DIR"), help="path to a directory with images"
    )
    parser.add_argument(
        "-i", "--images", default=2000, type=int, metavar="N", help="number of images for benchmarking (default: 2000)"
    )
    parser.add_argument(
        "-l", "--libraries", default=DEFAULT_BENCHMARKING_LIBRARIES, nargs="+", help="list of libraries to augbench"
    )
    parser.add_argument(
        "-r", "--runs", default=5, type=int, metavar="N", help="number of runs for each augbench (default: 5)"
    )
    parser.add_argument(
        "--show-std", dest="show_std", action="store_true", help="show standard deviation for augbench runs"
    )
    parser.add_argument("-p", "--print-package-versions", action="store_true", help="print versions of packages")
    parser.add_argument("-m", "--markdown", action="store_true", help="print benchmarking results as a markdown table")
    return parser.parse_args()
