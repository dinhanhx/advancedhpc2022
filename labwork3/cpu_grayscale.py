import numpy as np
import timeit
import skimage.io as skio

def rgb2gray(src, dst):
    for i in range(src.shape[0]):
        g = np.uint8((src[i, 0] + src[i, 1] + src[i, 2]) / 3)
        dst[i, 0] = dst[i, 1] = dst[i, 2] = g

dtime_list = []
for i in range(11):
    # Load and ignore alpha channel
    img = skio.imread('gura.png')[:, :, :3]
    h, w, _ = img.shape
    pixels_count = h * w
    img = np.reshape(img, (pixels_count, 3))
    gray_img = np.array(img, copy=True)

    # Measure time 
    stime = timeit.default_timer()
    rgb2gray(img, gray_img)
    dtime = timeit.default_timer() - stime
    dtime_list.append(dtime)

    gray_img = np.reshape(gray_img, (h, w, 3))
    skio.imsave('gray_gura.png', gray_img)
    
print(sum(dtime_list[1:])/len(dtime_list[1:]))