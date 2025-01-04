import os.path as osp
import cv2
import numpy as np
import matplotlib.pyplot as plt


def gaussian_filter(img, kernel_size, sigma):
    """Returns the image after Gaussian filter.
    Args:
        img: the input image to be Gaussian filtered.
        kernel_size: the kernel size in both the X and Y directions.
        sigma: the standard deviation in both the X and Y directions.
    Returns:
        res_img: the output image after Gaussian filter.
    """
    # create the Gaussian kernel
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    for x in range(kernel_size):
        for y in range(kernel_size):
            kernel[x, y] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(
                -((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
    kernel = kernel / np.sum(kernel)

    # apply the Gaussian filter
    rows, cols, channels = img.shape
    padded_img = np.pad(img, ((kernel_size // 2, kernel_size // 2),
                              (kernel_size // 2, kernel_size // 2), (0, 0)),
                        mode='constant', constant_values=0)
    res_img = np.zeros_like(img)

    for i in range(rows):
        for j in range(cols):
            for c in range(channels):
                # extract the neighborhood
                neighborhood = padded_img[i:i + kernel_size, j:j + kernel_size, c]
                # apply the kernel
                res_img[i, j, c] = np.sum(neighborhood * kernel)

    return res_img


if __name__ == "__main__":
    root_dir = osp.dirname(osp.abspath(__file__))
    img = cv2.imread(osp.join(root_dir, "Lena-RGB.jpg"))

    # parameters for the Gaussian filter
    parameters = [
        (5, 1),  # k = 5, sigma = 1
        (5, 3),  # k = 5, sigma = 3
        (7, 3)  # k = 7, sigma = 3
    ]

    # create a figure and a set of subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 5))

    # apply Gaussian filter for each set of parameters and display in subplots
    axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Input')

    for i, (kernel_size, sigma) in enumerate(parameters, start=1):
        res_img = gaussian_filter(img, kernel_size, sigma)
        axs[i // 2, i % 2].imshow(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB))
        axs[i // 2, i % 2].set_title(f'k = {kernel_size}, σ = {sigma}')

    plt.tight_layout()
    plt.show()
