"""The ACNe/CNe networks."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fcnet import FcNet
from models.nn_pts import Mlps


class Acn(nn.Module):
    """Attentive context normalization (ACN) layer."""

    def __init__(self, inc, atten_opt="softmax", eps=1e-3):
        """Initialize ACN with hyperparameters.

        Args:
            inc: input dimension.
            atten_opt: attention options. When attention_opt='None'`,
                we don't use attention in normalization. The ACN becomes
                CN layer.
            eps: small epsilon to avoid division by zero.
        """
        super(Acn, self).__init__()
        self.atten_opt = atten_opt
        self.eps = eps  # epsilon to avoid division by zero.

        if self.atten_opt == "softmax":
            # Note that we use a 1x1 convolution to convert the features we
            # have in the current feature map into attention (single channel
            # output)
            self.att_layer = nn.Conv2d(inc, 1, 1)
        else:
            pass  # No layer required for CN layer.

    def forward(self, x):
        b, _, n, _ = x.shape

        # Calculate the attention w.r.t the `self.atten_opt`. For this
        # assignment we only use the method where we use a single linear layer
        # to convert features into attention. In the ACNe paper, there's more!
        if self.atten_opt == "softmax":
            # TODO: (5 points) Invoke the attention layer on data
            # `self.attn_layer`, which will now give you a tensor of shape (b,
            # 1, n, 1). Then apply a soft max on the third dimension (dim=2) to
            # turn that into an attention vector that sums to 1 in that
            # dimension. We will then use this value to do a weighted mean and
            # standard deviation computation below.
            x_att = self.att_layer(x)
            self.softmax_layer = nn.Softmax(dim=2)
            a = self.softmax_layer(x_att)
        else:
            # Note that below is the case where we simply do averaging without
            # attention.
            a = torch.ones((b, 1, n, 1), dtype=torch.float32, device=x.device)
            a = a / a.sum(dim=2, keepdim=True)

        # TODO: (5 points) Calculate the weighted statistics -- mean and std.
        # Mean/Std should be of shape (b,c,1,1), which you then apply to your
        # data to make mean zero std 1.
        weighted_x = a * x
        weighted_mean = torch.mean(weighted_x, dim=2, keepdim=True)
        weighted_std = torch.std(weighted_x, dim=2, keepdim=True, correction=0)

        out = (x - weighted_mean) / (weighted_std + self.eps)

        return out


class AcneBlock(nn.Module):
    """ACNe Block.

    Each ACNe block is composed of linear layer, ACN layer,
    batch normalization, and activation layer.
    """

    def __init__(self, inc, outc, cn_opt="acn"):
        """ """
        super().__init__()
        if inc != outc:
            self.pass_layer = Mlps(inc, [outc])
            inc = outc
        else:
            self.pass_layer = nn.Identity()

        atten_opt = "softmax" if cn_opt == "acn" else "None"

        self.layer = nn.Sequential()
        # TODO: (5 points) Compose the main layers into a ACNe block. An ACNe block
        # consists of a "linear" layer which is a `nn.Conv2d` with 1x1 kernel
        # size and input channel size of `inc`, output channel size of `outc`,
        # followed by an "acn" layer which is the `Acn` block defined above,
        # followed by a "bn" layer that is `nn.BatchNorm2D` block, and finally
        # a "relu" layer which is using `nn.ReLU`.
        self.layer.add_module(f"Linear-{0}", nn.Conv2d(inc, outc, 1))
        self.layer.add_module(f"acn-{0}", Acn(outc, cn_opt))
        self.layer.add_module(f"bn-{0}", nn.BatchNorm2d(outc))
        self.layer.add_module(f"relu-{0}", nn.ReLU(inplace=True))

    def forward(self, x):
        """Forward pass.

        Args:
            x: bcn1
        """
        x = self.pass_layer(x)

        # Residual connection
        out = x + self.layer(x)
        return out


class Acne(FcNet):
    """ACNe/CNe architecture."""

    def __init__(self, config):
        """Initialize network with hyperparameters.

        Args:
            config (ml_collections.dict): configuration hyperparameters.
        """
        super(Acne, self).__init__(config)
        self.config = config
        num_classes = config.num_classes

        outc_list_in = [32]
        self.in_layer = Mlps(config.indim, outc_list_in, last_bn_norm=True)

        # ACNe layers.
        self.acne_layers = nn.Sequential()
        num_acne_block = 2  # We use 2 for faster training.
        inc = outc_list_in[-1]
        outc = 32
        for i in range(num_acne_block):
            self.acne_layers.add_module(
                f"acne-{i}", AcneBlock(inc, outc, config.cn_opt)
            )
            inc = outc

        # output layer that calculate class scores.
        self.output_layer = nn.Linear(inc, num_classes)

    def forward(self, x):
        """Forward pass.

        Args:
            x (array): BxNx2, input tensor.
        """
        x = x.transpose(2, 1)
        x = self.in_layer(x, format="BCN")
        x = self.acne_layers(x.unsqueeze(-1)).squeeze(-1)
        x, _ = x.max(dim=2)
        logits = self.output_layer(x)

        return F.log_softmax(logits, dim=1)
