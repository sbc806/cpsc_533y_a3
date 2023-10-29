"""The very Naive model based of FC layers."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FcNet(nn.Module):
    """Simple network consisting of FC layers"""

    def __init__(self, config):
        """Initialize network with hyperparameters.

        Args:
            inc (int): number of channels in the input.
            config (ml_collections.dict): configuration hyperparameters.
        """
        super(FcNet, self).__init__()
        inc = config.num_pts * 2
        self.config = config
        num_classes = config.num_classes
        outc_list = config.outc_list

        # Compose the model according to configuration specs
        self.net = nn.Sequential()
        for i, outc in enumerate(outc_list):
            self.net.add_module(f"Linear-{i}", nn.Linear(inc, outc))
            self.net.add_module(f"ReLU-{i}", nn.ReLU(inplace=True))
            inc = outc
        self.net.add_module(f"output", nn.Linear(inc, num_classes))

    def forward(self, x):
        """Forward pass.

        Args:
            x (array): BxNx2, input tensor.
        """
        if self.config.order_pts:
            _, indices = torch.sort(x[:, :, :1], axis=1)
            indices = indices.expand(-1, -1, x.shape[-1])
            x = torch.gather(x, 1, indices)

        # Define the forward  pass and get the logits for classification.
        x = x.reshape(x.shape[0], -1)
        logits = self.net(x)

        return F.log_softmax(logits, dim=1)

    def get_loss(self, pred, label):
        """Compute loss by comparing prediction and labels.

        Args:
            pred (array): BxD, probability distribution over D classes.
            label (array): B, category label.
        Returns:
            loss (tensor): scalar, cross entropy loss for classfication.
        """
        loss = F.nll_loss(pred, label)
        return loss

    def get_acc(self, pred, label):
        """Compute the acccuracy."""
        pred_choice = pred.max(dim=1)[1]
        acc = (pred_choice == label).float().mean()
        return acc
