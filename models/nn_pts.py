"""Common utility modules and functions for point-based network."""
from typing import ForwardRef

import torch
import torch.nn as nn


class Mlps(nn.Module):
    """Mlps implemented as (1x1) convolution."""

    def __init__(self, inc, outc_list=[128], last_bn_norm=True):
        """Initialize network with hyperparameters.

        Args:
            inc (int): number of channels in the input.
            outc_list (List[]): list of dimensions of hidden layers.
            last_bn_norm (boolean): determine if bn and norm layer is added into the output layer.
        """
        assert len(outc_list) > 0
        super(Mlps, self).__init__()

        self.layers = nn.Sequential()

        # We compose MLPs according to the list of out_channel (`outc_list`).
        # Additionally, we use the flag `last_bn_norm` to
        # determine if we want to add norm and activation layers
        # at last layer.
        for i, outc in enumerate(outc_list):
            self.layers.add_module(f"Linear-{i}", nn.Conv2d(inc, outc, 1))
            if i + 1 < len(outc_list) or last_bn_norm:
                self.layers.add_module(f"BN-{i}", nn.BatchNorm2d(outc))
                self.layers.add_module(f"ReLU-{i}", nn.ReLU(inplace=True))
            inc = outc

    def forward(self, x, format="BCNM"):
        """Forward pass.

        Args:
            x (torch.tensor): input tensor.
            format (str): format of point tensor.
                Options include 'BCNM', 'BNC', 'BCN'
        """
        assert format in ["BNC", "BCNM", "BCN"]

        # Re-formate tensor into "BCNM".
        if format == "BNC":
            x = x.transpose(2, 1).unsqueeze(-1)
        elif format == "BCN":
            x = x.unsqueeze(-1)
        
        # We use the tensor of the "BCNM" format.
        x = self.layers(x)

        # Re-formate tensor back input format.
        if format == "BNC":
            x = x.squeeze(-1).transpose(2, 1)
        elif format == "BCN":
            x = x.squeeze(-1)

        return x


def get_knn_idx(p1, p2, k):
    """Get index of k points of p2 nearest to p1.

    Args:
        p1 (tensor): a batch of point sets with shape of `(b, c, m)`
        p2 (tensor): a batch of point sets with shape of `(b, c, n)`
        k: the number of neighboring points.
    Returns:
        idx (tensor): the index of neighboring points w.r.t p1 in p2
            with shape of `(b, m, k)`.
    """
    # TODO: (10 points) Return the index of the top k elements. Use
    # `torch.topk` and the `pairwise_sqrdist_b` function below.
    #
    # HINT: your intermediate distance array should be of shape (b, m, n) and
    # your index array of shape (b, m, k)
    #
    
    distances = pairwise_sqrdist_b(p1, p2)
    print(p1.shape, p2.shape)
    print(distances.shape)
    sorted_distances, indices = torch.sort(distances, dim=2)
    idx = indices[:, :, 0:k]
    print(idx.shape)
    return idx


# BELOW are functions provided for your convenience.
def pairwise_sqrdist_b(p, q):
    """Pairwise square distance between two point sets (Batched).

    We implement the memory efficient way to pair-wise distance vis refactorization:
        `(p - q)**2 = p**2 + q**2  - 2*p^T*p`

    Args:
        p (tensor): a batch of point sets with shape of `(b, c, m)`
        q (tensor): a batch of point sets with shape of `(b, c, n)`

    Returns:
        dist (tensor):  pairwise distance matrix.
    """
    dist = -2 * torch.matmul(p.transpose(2, 1), q)  # bmn
    p_sqr = (p ** 2).sum(dim=1, keepdim=True).transpose(2, 1)  # bm1
    q_sqr = (q ** 2).sum(dim=1, keepdim=True)  # b1n
    dist += p_sqr + q_sqr
    return dist


# Modified from: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/e365b9f7b9c3d7d6444278d92e298e3f078794e1/models/pointnet2_utils.py#L63
def sampling_fps(xyz, npoint):
    """
    Input:
        xyz (tensor): pointcloud coordinates data with shape of (b, d, n). 
        npoint (tensor): number of samples
    Return:
        sampled_pts (tensor): sampled pointcloud index, (b, d, m) 
    """
    xyz = xyz.transpose(2, 1)

    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    sampled_pts = index_points(xyz.transpose(2, 1), centroids)
    return sampled_pts



def index_points(points, idx):
    """
    Input:
        points (tensor): input points data of shape (b, c, n), 
        idx (tensor): sample index data, [b, s1, s2 ...].
    Return:
        new_points (tensor): indexed points data, [b, c, s1, s2 ..]
    """
    points = points.transpose(2, 1)
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :].moveaxis(-1, 1)
    return new_points
