import os.path as osp
import cv2
import numpy as np
import matplotlib.pyplot as plt


def histogram_equalization(img):
    """Returns the image after histogram equalization.
    Args:
        img: the input image to be executed for histogram equalization.
    Returns:
        res_img: the output image after histogram equalization.
    """
    # calculate the histogram of the image
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    # calculate the cumulative distribution function (CDF)
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    # use the linear interpolation of the CDF to find new pixel values
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    # apply the CDF to the image
    res_img = cdf[img]

    return res_img


def local_histogram_equalization(img):
    """Returns the image after local histogram equalization.
    Args:
        img: the input image to be executed for local histogram equalization.
    Returns:
        res_img: the output image after local histogram equalization.
    """
    # calculate the radius of the block
    block_size = 64
    radius = block_size // 2

    # create an output image with the same size as the input image
    res_img = np.zeros_like(img)

    # loop over each pixel in the image
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # define the block's boundaries
            start_x = max(j - radius, 0)
            end_x = min(j + radius + 1, img.shape[1])
            start_y = max(i - radius, 0)
            end_y = min(i + radius + 1, img.shape[0])

            # extract the block
            block = img[start_y:end_y, start_x:end_x]

            # calculate the histogram of the block
            hist, bins = np.histogram(block.flatten(), 256, [0, 256])

            # calculate the cumulative distribution function (CDF)
            cdf = hist.cumsum()
            cdf_normalized = cdf * 255 / cdf[-1]

            # use the linear interpolation of the CDF to find new pixel values
            cdf_m = np.ma.masked_equal(cdf, 0)
            cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
            cdf = np.ma.filled(cdf_m, 0).astype('uint8')

            # map the pixel values in the block using the CDF
            equalized_block = cdf[block]

            # place the transformed pixel value back into the output image
            res_img[i, j] = equalized_block[i - start_y, j - start_x]

    return res_img


def plot_image_with_histogram(image_path):
    # read the example image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # check if image is loaded properly
    if img is None:
        print("Error: Image could not be loaded.")
        return

    # calculate the histogram
    hist = cv2.calcHist([img], [0], None,
                        [256], [0, 256]).flatten()  # flatten the histogram

    # calculate the cumulative distribution function (CDF)
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    # create a figure with a subplot for the image and a subplot for the histogram
    plt.figure(figsize=(10, 5))

    # plot the image
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Image')
    plt.axis('off')  # turn off axis numbers and ticks

    # plot the histogram
    plt.subplot(1, 2, 2)
    plt.plot(hist, drawstyle='steps-post')
    plt.fill_between(range(256), hist, step='post', alpha=0.7)
    plt.plot(cdf_normalized, color='red')
    plt.title('Pixel Intensity Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.ylim([0, np.max(hist) + 1])  # set the y-axis limit to the maximum frequency

    # show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    root_dir = osp.dirname(osp.abspath(__file__))
    img = cv2.imread(osp.join(root_dir, "moon.png"), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    res_hist_equalization = histogram_equalization(img)
    res_local_hist_equalization = local_histogram_equalization(img)

    cv2.imwrite(osp.join(root_dir, "HistEqualization.jpg"), res_hist_equalization)
    cv2.imwrite(
        osp.join(root_dir, "LocalHistEqualization.jpg"), res_local_hist_equalization
    )

    plot_image_with_histogram('moon.png')
    plot_image_with_histogram('HistEqualization.jpg')
    plot_image_with_histogram('LocalHistEqualization.jpg')