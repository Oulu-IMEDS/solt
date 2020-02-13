from PIL import Image
import cv2
import argparse
import os
import pkg_resources
import sys
import math
import numpy as np
from augbench.constants import DEFAULT_BENCHMARKING_LIBRARIES
from pytablewriter import MarkdownTableWriter
from pytablewriter.style import Style


def read_img_pillow(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        img = img.resize((256, 256))
        return img.convert("RGB")


def read_img_cv2(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
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


def get_package_versions():
    packages = [
        "albumentations",
        "imgaug",
        "torchvision",
        "numpy",
        "opencv-python",
        "scikit-image",
        "scipy",
        "pillow",
        "pillow-simd",
        "augmentor",
        "solt",
    ]
    package_versions = {"Python": sys.version}
    for package in packages:
        try:
            package_versions[package] = pkg_resources.get_distribution(package).version
        except pkg_resources.DistributionNotFound:
            pass
    return package_versions


def format_results(images_per_second_for_aug, show_std=False):
    if images_per_second_for_aug is None:
        return "-"
    result = str(math.floor(np.mean(images_per_second_for_aug)))
    if show_std:
        result += " ± {}".format(math.ceil(np.std(images_per_second_for_aug)))
    return result


class MarkdownGenerator:
    def __init__(self, df, package_versions):
        self._df = df
        self._package_versions = package_versions
        self._libraries_description = {"torchvision": "(Pillow-SIMD backend)"}

    def _highlight_best_result(self, results):
        best_result = float("-inf")
        for result in results:
            try:
                result = int(result)
            except ValueError:
                continue
            if result > best_result:
                best_result = result
        return ["**{}**".format(r) if r == str(best_result) else r for r in results]

    def _make_headers(self):
        libraries = self._df.columns.to_list()
        columns = []
        for library in libraries:
            version = self._package_versions[library]
            library_description = self._libraries_description.get(library)
            if library_description:
                library += " {}".format(library_description)

            columns.append("{library}<br><small>{version}</small>".format(library=library, version=version))
        return [""] + columns

    def _make_value_matrix(self):
        index = self._df.index.tolist()
        values = self._df.values.tolist()
        value_matrix = []
        for transform, results in zip(index, values):
            row = [transform] + self._highlight_best_result(results)
            value_matrix.append(row)
        return value_matrix

    def _make_versions_text(self):
        libraries = ["Python", "numpy", "pillow-simd", "opencv-python", "scikit-image", "scipy"]
        libraries_with_versions = [
            "{library} {version}".format(library=library, version=self._package_versions[library].replace("\n", ""))
            for library in libraries
        ]
        return "Python and library versions: {}.".format(", ".join(libraries_with_versions))

    def print(self):
        writer = MarkdownTableWriter()
        writer.headers = self._make_headers()
        writer.value_matrix = self._make_value_matrix()
        writer.styles = [Style(align="left")] + [Style(align="center") for _ in range(len(writer.headers) - 1)]
        writer.write_table()
        print("\n" + self._make_versions_text())
