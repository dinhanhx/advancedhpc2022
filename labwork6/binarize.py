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
def binarize(src, dst, threshold):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    g = np.uint8((src[i, j, 0] + src[i, j, 1] + src[i, j, 2]) / 3)
    if g >= threshold:
        dst[i, j, 0] = dst[i, j, 1] = dst[i, j, 2] = 255
    elif g < threshold:
        dst[i, j, 0] = dst[i, j, 1] = dst[i, j, 2] = 0
        

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
        out = np.array(img, copy=True)
        
        # Send to GPU
        A = cuda.to_device(img)
        B = cuda.to_device(out)

        # Configure Cuda blocks
        grid_size = dual_tuple_division(block_size, (h, w))

        # Measure time 
        stime = timeit.default_timer()
        binarize[grid_size, block_size](A, B, 128)
        dtime = timeit.default_timer() - stime
        dtime_list.append(dtime)
        
        out = B.copy_to_host()
        skio.imsave('binarize_gura.png', out)

    avgtime = sum(dtime_list[1:])/len(dtime_list[1:])
    avgtime_list.append(avgtime)
    print(f'{avgtime} @ {block_size}')
