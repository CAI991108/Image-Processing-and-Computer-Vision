import os
import os.path as osp
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from dataset import KITTIDataset, PatchProvider
from siamese_neural_network import StereoMatchingNetwork, calculate_similarity_score


def hinge_loss(score_pos, score_neg, labels):
    """
    Computes the hinge loss and accuracy.

    Args:
        score_pos (torch.Tensor): similarity scores for positive pairs
        score_neg (torch.Tensor): similarity scores for negative pairs
        labels (torch.Tensor): ground truth labels

    Returns:
        loss (torch.Tensor): computed hinge loss
        acc (float): accuracy of the predictions
    """
    margin = 0.2
    # Compute the hinge loss
    loss = torch.mean(torch.clamp(margin - score_pos + score_neg, min=0))

    # Ensure score_pos and score_neg are 1D tensors
    # score_pos = score_pos.view(-1, score_pos.size(-1))  # Reshape to (batch_size * height, width)
    # score_neg = score_neg.view(-1, score_neg.size(-1))  # Reshape to (batch_size * height, width)

    # Aggregate score_pos and score_neg to match the size of labels
    score_pos = score_pos.mean(dim=1)  # Average over the height dimension
    score_neg = score_neg.mean(dim=1)  # Average over the height dimension

    # Create a tensor to store the predictions (1 for positive, 0 for negative)
    predictions = (score_pos > score_neg).float()
    predictions = predictions.view(-1, 576)
    predictions = predictions.mean(dim=1)

    # Ensure labels is 1D tensor with the same size as predictions
    # labels = labels.view(-1)  # Ensure labels is also 1D

    # Compute the accuracy
    acc = predictions.float().mean().item()

    return loss, acc


def training_loop(
        infer_similarity_metric,
        patches,
        optimizer,
        out_dir,
        iterations=100,
        batch_size=128,
):
    """
    Runs the training loop of the siamese network.

    Args:
        infer_similarity_metric (obj): pytorch module
        patches (obj): patch provider object
        optimizer (obj): optimizer object
        out_dir (str): output file directory
        iterations (int): number of iterations to perform
        batch_size (int): batch size
    """

    for i in range(iterations):
        # Get a batch of patches
        ref_patches, pos_patches, neg_patches = next(patches.iterate_batches(batch_size))

        # Move data to the appropriate device (GPU if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ref_patches = ref_patches.to(device)
        pos_patches = pos_patches.to(device)
        neg_patches = neg_patches.to(device)

        # Forward pass
        score_pos = calculate_similarity_score(infer_similarity_metric, ref_patches, pos_patches)
        score_neg = calculate_similarity_score(infer_similarity_metric, ref_patches, neg_patches)

        # Create labels
        labels = torch.ones(score_pos.size(0), device=device)  # Positive examples have a label of 1

        # Calculate loss and accuracy
        loss, acc = hinge_loss(score_pos, score_neg, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss and accuracy
        if i % 5 == 0:
            print(f"Iteration {i}, Loss: {loss.item()}, Accuracy: {acc}")

        # Save model checkpoint periodically
        if i % 20 == 0:  # Save checkpoint every 200 iterations
            checkpoint_path = osp.join(out_dir, f"model_checkpoint_{i}.pth")
            torch.save(infer_similarity_metric.state_dict(), checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")


def main():
    writer = SummaryWriter('runs/siamese_experiment_1')

    # Fix random seed for reproducibility
    np.random.seed(7)
    torch.manual_seed(7)

    # Hyperparameters
    training_iterations = 1000
    batch_size = 128
    learning_rate = 3e-4
    patch_size = 9

    # Shortcuts for directories
    root_dir = osp.dirname(osp.abspath(__file__))
    data_dir = osp.join(root_dir, "KITTI_2015_subset")
    out_dir = osp.join(root_dir, "output/siamese_network")
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    # Create dataloader for KITTI training set
    dataset = KITTIDataset(
        osp.join(data_dir, "training"),
        osp.join(data_dir, "training/disp_noc_0"),
    )
    # Load patch provider
    patches = PatchProvider(dataset, patch_size=(patch_size, patch_size))

    # Initialize the network
    infer_similarity_metric = StereoMatchingNetwork()
    # Set to train
    infer_similarity_metric.train()
    # uncomment if you don't have a gpu
    # infer_similarity_metric.to('cpu')
    optimizer = torch.optim.SGD(
        infer_similarity_metric.parameters(), lr=learning_rate, momentum=0.9
    )

    # Start training loop
    training_loop(
        infer_similarity_metric,
        patches,
        optimizer,
        out_dir,
        iterations=training_iterations,
        batch_size=batch_size,
    )

    for i in range(training_iterations):

        writer.add_scalar('Loss/train', loss.item(), i)
        writer.add_scalar('Accuracy/train', acc, i)

        for tag, value in infer_similarity_metric.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram(tag, value.data.cpu().numpy(), i)
            writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), i)

    writer.close()


if __name__ == "__main__":
    main()
