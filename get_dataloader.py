import functools
from os import replace

import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader, Subset

from utils import mnist_helper


def get_dataloader(config, mode):
    """Set up input pipeline and get dataloader."""
    assert mode in ["train", "test"]
    dataset = MnistptsDataset(config, mode)
    loader = functools.partial(
        DataLoader,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    loader_list = []

    # Use PyTorch Subset module to divide the training dataset into train
    # and validation. To do this, first find out how many items should go into
    # training and how many into validation using the configuration. Then,
    # shuffle the dataset randomly, and create two `Subset`s. Note that the
    # training dataset should be shuffled randomly everytime you go through the
    # dataset, while the validation set does not need to be.
    if mode == "train":
        ratio_tr_data = config.ratio_tr_data
        num_all = len(dataset)

        idx = np.random.permutation(np.arange(num_all))
        num_tr = int(ratio_tr_data * num_all)

        dataset_tr = Subset(dataset, idx[:num_tr])
        dataset_va = Subset(dataset, idx[num_tr:])

        loader_tr = loader(dataset=dataset_tr, shuffle=True)
        loader_va = loader(dataset=dataset_va, shuffle=False)

        loader_list += [loader_tr, loader_va]
        print(f"Number of training samples: {num_tr}")
        print(f"Number of valid samples: {num_all - num_tr}")
    elif mode == "test":

        num_all = len(dataset)
        loader_te = loader(
            dataset=dataset,
            shuffle=False,
        )

        loader_list += [loader_te]
        print(f"Number of test samples: {num_all}")
    else:
        raise NotImplementedError

    return loader_list


class MnistptsDataset(data.Dataset):
    """Dataset for Mnist point clouds."""

    def __init__(self, config, mode):
        """Define immutable variables for multi-threaded loading.

        Args:
            config (config_dict):  hyperparamter configuration.
            mode: type of datset split.
        """

        assert mode in ["train", "test"]

        self.mode = mode
        self.config = config
        self.num_pts = config.num_pts
        print(f"loading {mode} datasets.")

        # Our dataset is small. Load the entire dataset into memory to
        # avoid excessive disk access!
        self.pts_list, self.labels = mnist_helper.load_mnistpts(
            config.data_mnistpts_dir, mode=mode
        )

    def __len__(self):
        """Return the length of dataset."""
        # return the length of dataset.
        return len(self.pts_list)

    def random_sampling(self, pts, num_pts):
        """Sampling points from point cloud.

        Args:
            pts (array): Nx2, point cloud.
        Returns:
            pts_sampled (array):  num_ptsX2, sampled point cloud.
        """

        # Sample points from point cloud. Importantly, we will sample
        # **without** replacement here to simulate how many actual point cloud
        # data behaves.
        #
        # Note: Random state might improperly be shared among threads. This is
        #   especially true if you use numpy to sample. Use PyTorch!
        idx = torch.multinomial(
            torch.ones(len(pts)), num_pts, replacement=False
        )
        pts_sampled = pts[idx]

        return pts_sampled

    def __getitem__(self, index):
        """Get item"""

        # get item from dataset.
        # Note that we expect: pc (np.float32 type), label(np.int)
        pc = np.asarray(self.pts_list[index]).astype("float32")
        label = np.asarray(self.labels[index]).astype("int")
        data = {
            "pc": self.random_sampling(pc, self.num_pts),
            "label": label,
        }

        return data
