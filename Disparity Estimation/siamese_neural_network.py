import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StereoMatchingNetwork(torch.nn.Module):
    """
    The network should consist of the following layers:
    - Conv2d(..., out_channels=64, kernel_size=3)
    - ReLU()
    - Conv2d(..., out_channels=64, kernel_size=3)
    - ReLU()
    - Conv2d(..., out_channels=64, kernel_size=3)
    - ReLU()
    - Conv2d(..., out_channels=64, kernel_size=3)
    - functional.normalize(..., dim=1, p=2)

    Remark: Note that the convolutional layers expect the data to have shape
        `batch size * channels * height * width`. Permute the input dimensions
        accordingly for the convolutions and remember to revert it before returning the features.
    """

    def __init__(self):
        """
        Implementation of the network architecture.
        Layer output tensor size: (batch_size, n_features, height - 8, width - 8)
        """

        super().__init__()
        gpu = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.to(gpu)  # move the model to the GPU or CPU

        # Change in_channels to 1 for grayscale images
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1).to(gpu)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1).to(gpu)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1).to(gpu)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1).to(gpu)
        self.norm = nn.functional.normalize

    def forward(self, X):
        """
        The forward pass of the network. Returns the features for a given image patch.

        Args:
            X (torch.Tensor): image patch of shape (batch_size, height, width, n_channels)

        Returns:
            features (torch.Tensor): predicted normalized features of the input image patch X,
                               shape (batch_size, height - 8, width - 8, n_features)
        """

        # ensure input is on the same device as the model
        X = X.to(self.conv1.weight.device)
        # permute the input dimensions for convolutions
        X = X.permute(0, 3, 1, 2)

        # apply convolutions and ReLU activations
        features = self.conv1(X)
        features = self.relu1(features)
        features = self.conv2(features)
        features = self.relu2(features)
        features = self.conv3(features)
        features = self.relu3(features)
        features = self.conv4(features)

        # normalize the features
        features = self.norm(features, dim=1, p=2)

        # revert the dimensions before returning
        features = features.permute(0, 2, 3, 1)
        return features


def calculate_similarity_score(infer_similarity_metric, Xl, Xr):
    """
    Computes the similarity score for two stereo image patches.

    Args:
        infer_similarity_metric (torch.nn.Module):  pytorch module object
        Xl (torch.Tensor): tensor holding the left image patch
        Xr (torch.Tensor): tensor holding the right image patch

    Returns:
        score (torch.Tensor): the similarity score of both image patches which is the dot product of their features
    """

    gpu = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    Xl = Xl.to(gpu)
    Xr = Xr.to(gpu)

    infer_similarity_metric.to(gpu)

    # Forward pass through the network to get features for both patches
    features_left = infer_similarity_metric(Xl)
    features_right = infer_similarity_metric(Xr)

    # Ensure that the features are 3D tensors before permuting
    features_left = features_left.view(features_left.size(0), -1, features_left.size(1))
    features_right = features_right.view(features_right.size(0), -1, features_right.size(1))

    # Compute the similarity score as the dot product of the features
    score = torch.bmm(features_left, features_right.permute(0, 2, 1))

    return score
