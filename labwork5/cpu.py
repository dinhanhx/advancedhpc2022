import numpy as np
import timeit
import skimage.io as skio
import math


def conv2d(src, dst, k):
    k_h, k_w, = k.shape
    src_h, src_w, _ = src.shape
    for c in range(3):
        for i in range(src_h):
            for j in range(src_w):
                temp = 0
                for m in range(-(k_h//2), k_h//2 + 1):
                    for n in range(-(k_w//2), k_w//2 + 1):
                        if i + m >= 0 and j + n >= 0 and i + m < src_h and j + n < src_w:
                            temp += np.float32(src[i+m, j+n, c]) * k[m + k_h // 2, n + k_w // 2]
                dst[i, j, c] = temp

# Gaussian Blur Kernel
gbk = np.array([[0, 0, 1, 2, 1, 0, 0],
                [0, 3, 13, 22, 13, 3, 0],
                [1, 13, 59, 97, 59, 13, 1],
                [2, 22, 97, 159, 97, 22, 2],
                [1, 13, 59, 97, 59, 13, 1],
                [0, 3, 13, 22, 13, 3, 0],
                [0, 0, 1, 2, 1, 0, 0]],
               dtype=np.float32) / 1003

# Load and ignore alpha channel
img = skio.imread('foxy_hmm.png')[:, :, :3]
img = np.ascontiguousarray(img)
h, w, _ = img.shape
out = np.array(img, copy=True)

conv2d(img, out, gbk)

skio.imsave('blur_foxy_hmm.png', out)