import numpy as np
import timeit
import skimage.io as skio
from numba import cuda
import math

def dual_tuple_division(x, y):
    return_tuple = []
    for i, ii in zip(x, y):
        return_tuple.append(math.ceil(ii/i))
    return tuple(return_tuple)

@cuda.jit
def blend(src, dst, coeff):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    for c in range(3):
        dst[i, j, c] = src[i, j, c] * coeff + (1 - coeff) * dst[i, j, c]

block_size_list = [(2,2),
                   (4, 4),
                   (8, 8),
                   (16, 16), 
                   (32, 32)]

avgtime_list = []
for block_size in block_size_list:
    dtime_list = []
    for i in range(11):
        # Load and ignore alpha channel
        img = skio.imread('gura.png')[:, :, :3]
        img = np.ascontiguousarray(img)
        h, w, _ = img.shape
        out = skio.imread('red_gura.png')[:, :, :3]
        out = np.ascontiguousarray(out)
        
        # Send to GPU
        A = cuda.to_device(img)
        B = cuda.to_device(out)

        # Configure Cuda blocks
        grid_size = dual_tuple_division(block_size, (h, w))

        # Measure time 
        stime = timeit.default_timer()
        blend[grid_size, block_size](A, B, 0.5)
        dtime = timeit.default_timer() - stime
        dtime_list.append(dtime)
        
        out = B.copy_to_host()
        skio.imsave('blend_gura.png', out)

    avgtime = sum(dtime_list[1:])/len(dtime_list[1:])
    avgtime_list.append(avgtime)
    print(f'{avgtime} @ {block_size}')
