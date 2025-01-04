import os
import os.path as osp

import numpy as np
from dataset import KITTIDataset
from matplotlib import pyplot as plt


def add_padding(I, padding):
    """
    Adds zero padding to an RGB or grayscale image.

    Args:
        I (np.ndarray): HxWx? numpy array containing RGB or grayscale image

    Returns:
        P (np.ndarray): (H+2*padding)x(W+2*padding)x? numpy array containing zero padded image
    """
    if len(I.shape) == 2:
        H, W = I.shape
        padded = np.zeros((H + 2 * padding, W + 2 * padding), dtype=np.float32)
        padded[padding:-padding, padding:-padding] = I
    else:
        H, W, C = I.shape
        padded = np.zeros((H + 2 * padding, W + 2 * padding, C), dtype=I.dtype)
        padded[padding:-padding, padding:-padding] = I

    return padded


def sad(image_left, image_right, window_size=3, max_disparity=50):
    """
    Compute the sum of absolute differences between image_left and image_right.

    Args:
        image_left (np.ndarray): HxW numpy array containing grayscale right image
        image_right (np.ndarray): HxW numpy array containing grayscale left image
        window_size: window size (default 3)
        max_disparity: maximal disparity to reduce search range (default 50)

    Returns:
        D (np.ndarray): HxW numpy array containing the disparity for each pixel
    """

    D = np.zeros_like(image_left)

    # add zero padding
    padding = window_size // 2
    image_left = add_padding(image_left, padding).astype(np.float32)
    image_right = add_padding(image_right, padding).astype(np.float32)

    height = image_left.shape[0]
    width = image_left.shape[1]

    # iterate through each pixel in the left image
    for i in range(padding, height - padding):
        for j in range(padding + max_disparity, width - padding):
            min_sad = float('inf')
            best_disparity = 0

            # get left patch
            left_patch = image_left[i - padding:i + padding + 1, j - padding:j + padding + 1]

            # compute SAD with patches in right image within disparity range
            for d in range(max_disparity):
                # ensure the disparity doesn't go out of the image bounds
                if j - d - padding < 0:
                    break

                # get right patch
                right_patch = image_right[i - padding:i + padding + 1, j - d - padding:j - d + padding + 1]

                # calculate SAD
                sad_value = np.sum(np.abs(left_patch - right_patch))

                # update the best disparity for minimum SAD
                if sad_value < min_sad:
                    min_sad = sad_value
                    best_disparity = d

            # assign best disparity to the disparity map
            D[i - padding, j - padding] = best_disparity

    return D


def visualize_disparity(
    disparity, im_left, im_right, out_file_path, title="Disparity Map", max_disparity=50
):
    """
    Generates a visualization for the disparity map.

    Args:
        disparity (np.array): disparity map
        im_left (np.array): left image
        im_right (np.array): right image
        out_file_path: output file path
        title: plot title
        max_disparity: maximum disparity
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.imshow(im_left, cmap='gray')
    plt.title('Example of Input Image')
    plt.axis('off')

    plt.subplot(2, 1, 2)
    plt.imshow(disparity, cmap='jet', vmin=0, vmax=max_disparity)
    plt.title(title)
    plt.colorbar(label='Disparity')
    plt.axis('off')

    plt.tight_layout()

    plt.savefig(out_file_path)
    plt.close()


def main():
    # Hyperparameters
    window_size = 15
    max_disparity = 50

    # Shortcuts
    root_dir = osp.dirname(osp.abspath(__file__))
    data_dir = osp.join(root_dir, "KITTI_2015_subset")
    out_dir = osp.join(
        root_dir, "output/handcrafted_stereo", f"window_size_{window_size}"
    )
    if not osp.isdir(out_dir):
        os.makedirs(out_dir)

    # Load dataset
    dataset = KITTIDataset(osp.join(data_dir, "testing"))

    # Calculation and Visualization
    for i in range(len(dataset)):
        # Load left and right images
        im_left, im_right = dataset[i]
        im_left, im_right = im_left.squeeze(-1), im_right.squeeze(-1)

        # Calculate disparity
        D = sad(im_left, im_right, window_size=window_size, max_disparity=max_disparity)

        # Define title and output file name for the plot
        title = f"Disparity map for image {i} with block matching (window size {window_size})"
        out_file_path = osp.join(out_dir, f"{i}_w{window_size}.png")

        # Visualize the disparity and save it to a file
        visualize_disparity(
            D,
            im_left,
            im_right,
            out_file_path,
            title=title,
            max_disparity=max_disparity,
        )


if __name__ == "__main__":
    main()
