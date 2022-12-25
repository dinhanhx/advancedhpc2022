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
def conv2d(src, dst, k):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    k_h, k_w, = k.shape
    src_h, src_w, _ = src.shape
    for c in range(3):
        temp = 0
        for m in range(-(k_h//2), k_h//2 + 1):
            for n in range(-(k_w//2), k_w//2 + 1):
                if i + m >= 0 and j + n >= 0 and i + m < src_h and j + n < src_w:
                    temp += np.float32(src[i+m, j+n, c]) * k[m + k_h // 2, n + k_w // 2]
        dst[i, j, c] = temp

block_size_list = [(2,2),
                   (4, 4),
                   (8, 8),
                   (16, 16), 
                   (32, 32),]

# Gaussian Blur Kernel
gbk = np.array([[0, 0, 1, 2, 1, 0, 0],
                [0, 3, 13, 22, 13, 3, 0],
                [1, 13, 59, 97, 59, 13, 1],
                [2, 22, 97, 159, 97, 22, 2],
                [1, 13, 59, 97, 59, 13, 1],
                [0, 3, 13, 22, 13, 3, 0],
                [0, 0, 1, 2, 1, 0, 0]],
               dtype=np.float32) / 1003

avgtime_list = []
for block_size in block_size_list:
    dtime_list = []
    for i in range(11):
        # Load and ignore alpha channel
        img = skio.imread('foxy_hmm.png')[:, :, :3]
        img = np.ascontiguousarray(img)
        h, w, _ = img.shape
        out = np.array(img, copy=True)
        
        # Send to GPU
        A = cuda.to_device(img)
        B = cuda.to_device(out)
        C = cuda.to_device(gbk)

        # Configure Cuda blocks
        grid_size = dual_tuple_division(block_size, (h, w))

        # Measure time 
        stime = timeit.default_timer()
        conv2d[grid_size, block_size](A, B, C)
        dtime = timeit.default_timer() - stime
        dtime_list.append(dtime)
        
        out = B.copy_to_host()
        skio.imsave('blur_foxy_hmm.png', out)

    avgtime = sum(dtime_list[1:])/len(dtime_list[1:])
    avgtime_list.append(avgtime)
    print(f'{avgtime} @ {block_size}')