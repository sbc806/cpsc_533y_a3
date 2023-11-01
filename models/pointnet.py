"""Pointnet vanilla."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fcnet import FcNet
from models.nn_pts import Mlps


class PointNet(FcNet):
    """PointNet vanilla."""

    def __init__(self, config):
        """Initialize network with hyperparameters.

        Args:
            config (ml_collections.dict): configuration hyperparameters.
        """
        super(PointNet, self).__init__(config)
        self.config = config
        num_classes = config.num_classes

        # TODO: (5 points) Use the class `Mlps` to implement the encoder part
        # of the pointNet. `last_bn_norm` should be True for our example, and
        # we would like to use 3 blocks, where each of our pointnet block to
        # have 32 neurons.
        self.net = nn.Sequential()
        inc = 2
        # self.encoder = Mlps(inc, [32, 32, 32], last_bn_norm=True)
        self.net.add_module(f"Mlps-{1}", Mlps(inc, [32], last_bn_norm=True))
        self.net.add_module(f"Mlps-{2}", Mlps(32, [32], last_bn_norm=True))
        self.net.add_module(f"Mlps-{3}", Mlps(32, [32], last_bn_norm=True))
        # TODO: (5 points) Implement the output layer that converts the
        # max-pooled global feature into `logits` to be used for
        # classification. In other words, it should be a simple linear layer
        # without any activation.
        self.output_layer = nn.Linear(32, num_classes)

    def forward(self, x):
        """Forward pass.

        Args:
            x (array): BxNx2, input tensor.
        """

        # TODO: (10 points) Implement the forward pass of a point net. We first encode
        # global features with our encoder, we then perform global max pooling,
        # and then apply the output layer. Note that when calling our
        # `self.encoder`, you might want to investigate the `format` option for
        # easy processing.
        encoded_x = self.net(x, format="BNC")
        print("encoded shape", encoded_x.shape)
        self.pooling_layer = nn.MaxPool1d(32)
        pooled_x = self.pooling_layer(encoded_x.transpose(2, 1)).squeeze(-1)
        logits = self.output_layer(pooled_x)
        # NOTE: we get logits as outputs, which we then use log_softmax to use
        # in our loss function.
        return F.log_softmax(logits, dim=1)
