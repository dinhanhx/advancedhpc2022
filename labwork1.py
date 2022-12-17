import numba
from numba import cuda

cuda.detect()

cuda.select_device(0)
