import numpy as np
import timeit
import skimage.io as skio
from numba import cuda
import math

@cuda.jit
def rgb2gray(src, dst):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    g = np.uint8((src[i, 0] + src[i, 1] + src[i, 2]) / 3)
    dst[i, 0] = dst[i, 1] = dst[i, 2] = g

block_size_list = [16, 32, 64, 128, 512, 1024]
avgtime_list = []
for block_size in block_size_list:
    dtime_list = []
    for i in range(11):
        # Load and ignore alpha channel
        img = skio.imread('gura.png')[:, :, :3]
        img = np.ascontiguousarray(img)
        h, w, _ = img.shape
        pixels_count = h * w
        img = np.reshape(img, (pixels_count, 3))
        A = cuda.to_device(img)
        gray_img = np.array(img, copy=True)
        B = cuda.to_device(gray_img)

        # Configure Cuda blocks
        grid_size = math.ceil(pixels_count/block_size)

        # Measure time 
        stime = timeit.default_timer()
        rgb2gray[grid_size, block_size](A, B)
        dtime = timeit.default_timer() - stime
        dtime_list.append(dtime)

        gray_img = np.reshape(B.copy_to_host(), (h, w, 3))
        skio.imsave('gray_gura.png', gray_img)

    avgtime = sum(dtime_list[1:])/len(dtime_list[1:])
    avgtime_list.append(avgtime)
    print(f'{avgtime} @ {block_size}')
    
import matplotlib.pyplot as plt
plt.plot(block_size_list, avgtime_list)