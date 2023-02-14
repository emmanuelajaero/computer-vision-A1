import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread

from utils import filter2d, gaussian_kernel


def main():

    # load the image
    im = imread("paint.jpg").astype("float")
    im = im / 255

    # number of levels for downsampling
    N_levels = 5

    # make a copy of the original image
    im_subsample = im.copy()

    # naive subsampling, visualize the results on the 1st row
    for i in range(N_levels):
        # subsample image
        im_subsample = im_subsample[::2, ::2, :]
        plt.subplot(2, N_levels, i + 1)
        plt.imshow(im_subsample)
        plt.axis("off")

    # subsampling without aliasing, visualize results on 2nd row
    kernel = gaussian_kernel()
    im_anti_alias = im.copy()
    for i in range(N_levels):
        im_anti_alias[:, :, 0] = filter2d(im_anti_alias[:, :, 0], kernel)
        im_anti_alias[:, :, 1] = filter2d(im_anti_alias[:, :, 1], kernel)
        im_anti_alias[:, :, 2] = filter2d(im_anti_alias[:, :, 2], kernel)

        # subsample image
        im_anti_alias = im_anti_alias[::2, ::2, :]
        plt.subplot(2, N_levels, N_levels + i + 1)
        plt.imshow(im_anti_alias)
        plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
