import matplotlib.pylab as plt
import numpy as np
from skimage import io

from utils import filter2d, gaussian_kernel, partial_x, partial_y


def main():
    # Load image
    img = io.imread("iguana.png", as_gray=True)

    # Smooth image with Gaussian kernel
    denoised_image = filter2d(img, gaussian_kernel())

    # Compute x and y derivate on smoothed image
    x_derivative = partial_x(denoised_image)
    y_derivative = partial_y(denoised_image)

    # Compute gradient magnitude
    gradient = x_derivative + y_derivative

    # Visualize results
    # fig, ax = plt.subplots()
    # ax.imshow(x_derivative, cmap="gray")
    # plt.show()

    rows = 3
    columns = 1

    fig = plt.figure(figsize=(10, 10))

    # Adds x derivative to subplot
    fig.add_subplot(rows, columns, 1)

    plt.imshow(x_derivative, cmap="gray")
    plt.axis("off")
    plt.title("x derivative")

    # Adds y derivative to subplot
    fig.add_subplot(rows, columns, 2)

    # showing image
    plt.imshow(y_derivative, cmap="gray")
    plt.axis("off")
    plt.title("y derivative")

    # Adds Gradient Magnitude to subplot
    fig.add_subplot(rows, columns, 3)

    # showing image
    plt.imshow(gradient, cmap="gray")
    plt.axis("off")
    plt.title("Gradient Magnitude")

    plt.show()


if __name__ == "__main__":
    main()
