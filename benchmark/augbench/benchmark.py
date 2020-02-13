import os

os.environ["OMP_NUM_THREADS"] = "1"  # noqa E402
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # noqa E402
os.environ["MKL_NUM_THREADS"] = "1"  # noqa E402
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # noqa E402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa E402

from augbench import utils

if __name__ == "__main__":
    args = utils.parse_args()
