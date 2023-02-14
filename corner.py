import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import peak_local_max
from skimage.io import imread

from utils import filter2d, gaussian_kernel, partial_x, partial_y


def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """
    gaussian_filter = gaussian_kernel(window_size, 1.0)
    response = None

    # Compute the derivatives using Sobel operator
    Ix = partial_x(img)
    Iy = partial_y(img)

    Ix_square_guassian = filter2d(Ix**2, gaussian_filter)
    Iy_square_guassian = filter2d(Iy**2, gaussian_filter)
    IxIy_guassian = filter2d(Ix * Iy, gaussian_filter)

    response = (Ix_square_guassian * Iy_square_guassian - IxIy_guassian**2) - k * (
        Ix_square_guassian + Iy_square_guassian
    ) ** 2

    return response


def main():
    img = imread("building.jpg", as_gray=True)

    # Compute Harris corner response
    response = harris_corners(img)

    # Threshold on response
    threshold = (response > 0.03125) * response

    # Perform non-max suppression by finding peak local maximum
    nms = peak_local_max(threshold)

    # Visualize results

    nms_figure = plt.figure("NMS")
    plt.imshow(img, cmap="gray")
    plt.scatter(nms[:, 1], nms[:, 0], marker="x")
    plt.title("NMS")
    plt.axis("off")

    threshold_figure = plt.figure("Threshold")
    plt.imshow(threshold, cmap="gray")
    plt.title("Threshold")
    plt.axis("off")

    response_figure = plt.figure("Response")
    plt.imshow(response, cmap="gray")
    plt.title("Response")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
