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
def rgb2gray(src, dst):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    ii = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    g = np.uint8((src[i, ii, 0] + src[i, ii, 1] + src[i, ii, 2]) / 3)
    dst[i, ii, 0] = dst[i, ii, 1] = dst[i, ii, 2] = g

block_size_list = [(2,2),
                   (4, 4),
                   (8, 8),
                   (16, 16), 
                   (32, 32),]
avgtime_list = []
for block_size in block_size_list:
    dtime_list = []
    for i in range(11):
        # Load and ignore alpha channel
        img = skio.imread('gura.png')[:, :, :3]
        img = np.ascontiguousarray(img)
        h, w, _ = img.shape
        A = cuda.to_device(img)
        gray_img = np.array(img, copy=True)
        B = cuda.to_device(gray_img)

        # Configure Cuda blocks
        grid_size = dual_tuple_division(block_size, (h, w))

        # Measure time 
        stime = timeit.default_timer()
        rgb2gray[grid_size, block_size](A, B)
        dtime = timeit.default_timer() - stime
        dtime_list.append(dtime)
        
        gray_img = B.copy_to_host()
        skio.imsave('gray_gura.png', gray_img)

    avgtime = sum(dtime_list[1:])/len(dtime_list[1:])
    avgtime_list.append(avgtime)
    print(f'{avgtime} @ {block_size}')