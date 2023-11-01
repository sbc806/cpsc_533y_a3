"""Simple implementation for Point Convolution.

Please note that this implementation differs from the original version Local neighborhood.
We use KNN in order to simplfy the assignment. 
"""
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fcnet import FcNet
from models.nn_pts import Mlps, get_knn_idx, index_points


class ConvLayer(nn.Module):
    """Set abstraction (main block in PointNet2) with pointnet."""

    # WEIWEI: above description seem wrong?
    def __init__(self, inc, outc, config):
        super().__init__()

        # TODO: We define the kernel generator.
        self.kernel_generator = Mlps(inc, [outc])

        self.k = config.k  # the size of neighborhood

    def forward(self, x):
        """Sample a subset from point cloud and learn features for the sampled points.

        Args:
            x (tuple): (feat, loc) where:
                feat(tensor): point feature of shape `(b, c, n)`.
                loc (tensor): point location of shape `(b, d, n)`.
        Returns:
            feat(tensor): output feature of shape `(b, c, n)` where c is 
                number of channels in output feature. 
        """
        feat, loc = x
        # Sample a set from input point cloud.
        b, c, n = feat.shape

        # NOTE: Steps 1 to 3 below are nearly identical for both pointnet2 and pointconv. 
        # The minor difference would be that pointnet2 operates on downsampled point clouds 
        # while pointconv operates on input resolution.

        # TODO: (0 points) Step 1 -- Use `get_knn_idx` to retrieve the index of the k
        # nearest neighbors in Euclidean space. It should be of shape (b, n,
        # k). The top-K should also consider itself.
        nearest_neighbors = get_knn_idx(loc, loc, self.k)

        # TODO: (0 points) Step 2 -- Use `index_points` to retrieve the features of these
        # k-nn points. Your retrieved features should be of (b, c, n, k) if
        # done properly.
        #
        retrieved_features = index_points(feat, nearest_neighbors)

        # TODO: (0 points) Step 3 -- Retrieve relative location of the top-K neighbors.
        # Note that you can again use `index_points` here. An easy way to
        # compute this would be to retrieve all top-k locations and subtract
        # the original location. Also, note that with numpy broadcasting, you
        # can easily do this. For example, if your top k locations are of shape
        # (b, d, n, k), you can subtract something of shape (b, d, n, 1) to get
        # the relative coordinates of all top-K samples for all samples.
        retrieved_locations = index_points(loc, nearest_neighbors)
        relative_locations = retrieved_locations - loc.unsqueeze(-1)

        # TODO: (5 points) Let's now regress the kernel from the coordinates
        # and apply them. Basically, use the `self.kernel_generator`, which is
        # an MLP and pass on the relative coordinates you retrieved in step 3
        # as input---as long as the input shape is (b,d,n,k) the function
        # should process the last two dimensions in the same way, and turn our
        # d dimension into kernel matrices of size (b, c1 x c2, n, k).
        kernel_matrix = self.kernel_generator(relative_locations)
        print(kernel_matrix.shape)

        # TODO: (10 points) Apply the kernel to our features obtained in step
        # 2. To do this you would need to reorder and reshape the kernel tensor
        # to be of shape (b, n, k, c1, c2) and also the features in step 2 into
        # shape that is (b, n, k, something, something). You can then treat
        # this as an array of matrices with `torch.matmul` to achieve a shape
        # of (b, n, k, c2, 1) or (b, n, k, 1, c2) at the end. You then need to
        # sum up along the `k` dimension here, ultimately achieving (b, n, c2)
        # as output.

        return feat.moveaxis(-1, 1)


class PointConv(FcNet):
    """Point Convolution."""

    def __init__(self, config):
        """Initialize network with hyperparameters.

        Args:
            config (ml_collections.dict): configuration hyperparameters.
        """
        super().__init__(config)
        self.config = config
        num_classes = config.num_classes

        self.in_layer = Mlps(config.indim, [32], last_bn_norm=True)

        # We currently use the single convolution layer for faster training.
        self.conv_layer = ConvLayer(inc=32, outc=32, config=config)

        # Output layers
        pointnet_mlp_outc_list = [32, 32]
        self.output_pointnet = Mlps(
            32, outc_list=pointnet_mlp_outc_list, last_bn_norm=True
        )
        self.output_layer = nn.Linear(pointnet_mlp_outc_list[-1], num_classes)

    def forward(self, x):
        """Forward pass.

        Args:
            x (array): BxNx2, input tensor.
        """
        x = x.transpose(2, 1)
        loc = x
        x = self.in_layer(x, format="BCN")
        x = self.conv_layer((x, loc))
        x = self.output_pointnet(x, format="BCN").max(dim=2)[0]
        logits = self.output_layer(x)

        return F.log_softmax(logits, dim=1)
