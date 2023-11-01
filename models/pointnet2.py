"""Simple implementation for PointNet++.

Please note that this implementation differs from the original version Local neighborhood.
We use KNN in order to simplfy the assignment. 
"""
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fcnet import FcNet
from models.nn_pts import Mlps, get_knn_idx, index_points, sampling_fps

import math


class SetAbstraction(nn.Module):
    """Set abstraction (main block in PointNet2) with pointnet."""

    def __init__(self, pointnet_mlp, config):
        super().__init__()
        self.config = config
        self.mlps = pointnet_mlp()
        self.downratio = config.downratio
        self.k = config.k  # the size of neighborhood

    def forward(self, x):
        """Sample a subset from point cloud and learn features for the sampled points.

        Args:
            x (tuple): (feat, loc) where:
                feat(tensor): point feature of shape `(b, c, n)`.
                loc (tensor): point location of shape `(b, d, n)`.
        Returns:
            (tuple): (set_feat, set_loc)
                set_feat: set feature (a subset of input point cloud) of shape `(b, c, m)`
                set_loc: set location of shape `(b, d, m)`
        """
        feat, loc = x
        b, c, n = feat.shape

        # TODO: (5 points) Sample a subset from the input point cloud using
        # `sampling_fps`. The number of samples you get should be based on
        # `self.downratio`. The example implementation uses flooring to when
        # the ratio converts the number of samples into floats. Our sampled
        # locations should be `set_loc` with size of m.
        m = math.floor(n * self.downratio)
        set_loc = sampling_fps(loc, m)
        print("n", n, "m", m)
        print("set_loc.shape", set_loc.shape)

        # NOTE: Steps 1 to 3 below are nearly identical for both pointnet2 and pointconv. 
        # The minor difference would be that pointnet2 operates on downsampled point clouds 
        # while pointconv operates on input resolution.

        # TODO: (5 points) Step 1 -- Use `get_knn_idx` to retrieve the index of the k
        # nearest neighbors in Euclidean space. It should be of shape (b, m,
        # k). The top-K should also consider itself.
        nearest_neighbors = get_knn_idx(set_loc, loc, self.k)
        print("nearest_neighbors.shape", nearest_neighbors.shape)

        # TODO: (5 points) Step 2 -- Use `index_points` to retrieve the features of these
        # k-nn points. Your retrieved features should be of (b, c, m, k) if
        # done properly.
        #
        retrieved_features = index_points(feat, nearest_neighbors)
        print("retrieved_features.shape", retrieved_features.shape)

        # TODO: (5 points) Step 3 -- Retrieve relative location of the top-K neighbors.
        # Note that you can again use `index_points` here. An easy way to
        # compute this would be to retrieve all top-k locations and subtract
        # the original location. Also, note that with numpy broadcasting, you
        # can easily do this. For example, if your top k locations are of shape
        # (b, d, m, k), you can subtract something of shape (b, d, m, 1) to get
        # the relative coordinates of all top-K samples for all samples.
        retrieved_locations = index_points(loc, nearest_neighbors)
        relative_locations = retrieved_locations - set_loc.unsqueeze(-1)
        print("relative_locations.shape", relative_locations.shape)

        # TODO: (5 points) Concatenate features from step 2 with their relative locations
        # from step 3. This should result in a (b, d+c, m, k) array.
        concatenated_features = torch.cat((retrieved_features, relative_locations), dim=1)
        print("concatenated_features.shape", concatenated_features.shape)

        # TODO: (5 points) Process concatenated features with `self.mlps` to obtain a new
        # set of features of shape (b, c', m, k) and then take the maximum
        # along the `k` dimension to get our final descriptors `set_feat`.
        processed_concatenated_features = self.mlps(concatenated_features)
        print("processed_concatenated_features.shape", processed_concatenated_features.shape)
        set_feat, _= torch.max(processed_concatenated_features, dim=-1)
        print("set_feat.shape", set_feat.shape)

        return (set_feat, set_loc)


class PointNet2(FcNet):
    """PointNet2 (i.e., PointNet++)."""

    def __init__(self, config):
        """Initialize network with hyperparameters.

        Args:
            config (ml_collections.dict): configuration hyperparameters.
        """
        super().__init__(config)
        self.config = config
        num_classes = config.num_classes

        inc = config.indim
        self.sa_layers = nn.Sequential()
        pointnet_mlp_outc_list = [32, 32]

        num_hierarchies = 1  # For faster training, we set it as 1.
        for i in range(num_hierarchies):
            pointnet_mlp = functools.partial(
                Mlps, inc=inc + config.indim, outc_list=pointnet_mlp_outc_list
            )

            self.sa_layers.add_module(
                f"SA-{i}", SetAbstraction(pointnet_mlp, config)
            )

            inc = pointnet_mlp_outc_list[-1]

        # Output layers that generate class scores.
        pointnet_mlp_outc_list = [32, 32, 32]
        self.output_pointnet = Mlps(
            inc + config.indim,
            outc_list=pointnet_mlp_outc_list,
            last_bn_norm=True,
        )
        self.output_layer = nn.Linear(pointnet_mlp_outc_list[-1], num_classes)

    def forward(self, x):
        """Forward pass.

        Args:
            x (array): (b, n, 2), input  point clouds.
        """
        x = x.transpose(2, 1)  # bdn
        feat, loc = self.sa_layers((x, x))

        # Global feature via PointNet.
        x = self.output_pointnet(
            torch.cat([feat, loc], dim=1), format="BCN"
        ).max(dim=2)[0]
        logits = self.output_layer(x)

        return F.log_softmax(logits, dim=1)
