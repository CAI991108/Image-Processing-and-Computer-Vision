# utils.py

# This file contains utility functions for the project.
# This file should include:
# - Loss computation (e.g., custom loss functions)
# - Metrics computation (e.g., IoU, mIoU)
# - Logging and visualization tools

import numpy as np
import scipy.misc
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int, long)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if not isinstance(alpha, list):
            pass
        else:
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input_loss, target):
        if input_loss.dim() > 2:
            input_loss = input_loss.view(input_loss.size(0), input_loss.size(1), -1)  # N,C,H,W => N,C,H*W
            input_loss = input_loss.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input_loss = input_loss.contiguous().view(-1, input_loss.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input_loss)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input_loss.data.type():
                self.alpha = self.alpha.type_as(input_loss.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields with the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()


if __name__ == '__main__':

    import model_rmppe as modellib
    import numpy as np

    logger = Logger('./logs')
    img = np.zeros((10, 3, 100, 100), dtype=np.uint8)

    print('===========> loading model <===========')
    netmodel = modellib.get_model()
    for tag, value in netmodel.named_parameters():
        print
        tag, value.data.cpu().numpy().shape

    print('===========> logger <===========')
    step = 0
    # (1) Log the scalar values
    info = {
        'loss': 0.5,
        'accuracy': 0.9
    }

    for tag, value in info.items():
        logger.scalar_summary(tag, value, step)

    # (2) Log values and gradients of the parameters (histogram)
    for tag, value in netmodel.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.data.cpu().numpy(), step)

    # (3) Log the images
    info = {
        'images': img
    }

    for tag, images in info.items():
        logger.image_summary(tag, images, step)


def calcIOU(img, mask):
    sum1 = img + mask
    sum1[sum1 > 0] = 1
    sum2 = img + mask
    sum2[sum2 < 2] = 0
    sum2[sum2 >= 2] = 1
    if np.sum(sum1) == 0:
        return 1
    else:
        return 1.0 * np.sum(sum2) / np.sum(sum1)


def test(dataLoader, netmodel, exp_args):
    # switch to eval mode
    netmodel.eval()
    softmax = nn.Softmax(dim=1)
    iou = 0
    for i, (input_ori, input, edge, mask) in enumerate(dataLoader):
        input_ori_var = Variable(input_ori.cuda())
        input_var = Variable(input.cuda())
        edge_var = Variable(edge.cuda())
        mask_var = Variable(mask.cuda())

        # compute output: loss part1
        if exp_args.addEdge == True:
            output_mask, output_edge = netmodel(input_ori_var)
        else:
            output_mask = netmodel(input_ori_var)

        prob = softmax(output_mask)[0, 1, :, :]
        pred = prob.data.cpu().numpy()
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        iou += calcIOU(pred, mask_var[0].data.cpu().numpy())

    print(len(dataLoader))
    return iou / len(dataLoader)