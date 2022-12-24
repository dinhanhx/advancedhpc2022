import numpy as np
import timeit
import skimage.io as skio

def rgb2gray(src, dst):
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            g = (int(src[i, j, 0]) + int(src[i, j, 1]) + int(src[i, j, 2])) / 3
            dst[i, j, 0] = dst[i, j, 1] = dst[i, j, 2] = g

dtime_list = []
for i in range(11):
    # Load and ignore alpha channel
    img = skio.imread('gura.png')[:, :, :3]
    img = np.ascontiguousarray(img)
    gray_img = np.array(img, copy=True)

    # Measure time 
    stime = timeit.default_timer()
    rgb2gray(img, gray_img)
    dtime = timeit.default_timer() - stime
    dtime_list.append(dtime)

    skio.imsave('gray_gura.png', gray_img)
    
print(sum(dtime_list[1:])/len(dtime_list[1:]))