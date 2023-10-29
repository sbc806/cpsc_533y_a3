"""Helper functions for processing MNIST data -- e.g., convert 2D image into point cloud."""
import glob
import matplotlib.pyplot as plt
from mnist.loader import MNIST
import numpy as np
import os
import os.path as osp
import pickle
import subprocess

def download_mnist(data_dir):
    """Download MNIST raw data."""
    if not osp.exists(data_dir):
        os.makedirs(data_dir)
    _DATA_URL = "http://yann.lecun.com/exdb/mnist/"
    cmd = ["wget"]
    cmd += ["--recursive", "--level=1", f"--directory-prefix={data_dir}"]
    cmd += ["--no-host-directories", "--cut-dirs=3"]
    cmd += ["--accept=*.gz", _DATA_URL]
    subprocess.run(cmd, capture_output=True)

def check_mnist(data_dir):
    """Check and dowload MNIST raw data."""
    gz_file = glob.glob(osp.join(data_dir, "*.gz"))
    downloaded = len(gz_file) == 4 
    if not downloaded:
        download_mnist(data_dir)
    else:
        print(f"MNIST found in {data_dir}")

def load_mnist(data_dir, mode="train"):
    """Load mnist dataset with the given mode.
    
    Args:
        data_dir (str): director to save MNIST raw data.
        mode (str): the type of data split.
    Returns:
        imgs_list (list[array]): List of image.
        labels (array): labels.
    """
    check_mnist(data_dir)
    data = MNIST(data_dir)
    data.gz = True
    if mode == "train":
        imgs_list, labels = data.load_training()
    elif mode == "test":
        imgs_list, labels = data.load_testing()
    else:
        raise NotImplementedError
    imgs_list = [np.array(img, dtype=np.uint8).reshape([28, 28], order="C") for img in imgs_list]
    
    return imgs_list, labels

def img2pts(imgs, num_pts=256):
    """Generate point clouds from mnist images.
    
    We convert MNIST's image into point cloud via binary thresholding on the density value. 
    During this processing, the balck background pixels have little information and thus are 
    removed. In this case, we obtain the coordinates of valid pixels which have the density
    large than a threshold.  

    Args:
        imgs (array): N images of size (28x28). 
        num_pts (int):  number of points sampled for each image. 
    Returns: 
        mnistpts (array): list of pts. 
    """
    mnistpts_list = []
    for img in imgs:
        img_norm = (img - img.min()) / (img.max() - img.min())
        mask = img_norm > 0.5

        # Image size is 28x28x2
        ii = np.arange(28)
        jj = np.arange(28)[::-1]
        indexs = np.stack(np.meshgrid(ii, jj, indexing="xy"), -1)
        pts = indexs[mask] 

        # Just in case that the number of valid pixel is small.   
        pts = np.concatenate([pts] * (num_pts // len(pts) + 1), 0)

        # Calculate the normalized coordinates.
        pts = (pts - 14.0) / 14.0
        mnistpts_list += [np.array(pts[:num_pts])]  # Get the uniform num of points
    return mnistpts_list 

def dump_mnistpts(data_mnist_dir, data_mnistpts_dir="data_dump", mode="train",):
    """Downloading and processing MNIST dataset.
    
    We first download MNIST image dataset into `data_mnist_dir` and then covert the image into point clouds.
    We currently save the resultant point cloud in `data_mnistpts_dir` as `.pkl` file.

    Args:
        data_mnist_dir (str): directory where we save the downloaded MNIST.
        data_mnistpts_dir (str): directory where we save the resultant point clouds.
    """ 
    imgs_list, labels = load_mnist(data_mnist_dir, mode) # 
    mnistpts = img2pts(imgs_list) # 
    dump_file_dir = osp.join(data_mnistpts_dir, mode)
    if not osp.exists(dump_file_dir):
        os.makedirs(dump_file_dir)

    dump_file_pts = osp.join(dump_file_dir, "pts.pkl")
    dump_file_labels = osp.join(dump_file_dir, "label.pkl")

    with open(dump_file_pts, "wb") as f:
        pickle.dump(mnistpts, f)
    with open(dump_file_labels, "wb") as f:
        pickle.dump(labels, f)

def load_mnistpts(dump_file_dir, mode="train"):
    """Load pts from `dump_file_dir`.

    Args:
        dump_file_dir (str): directory where dataset is saved.
        mode (str): type of data split.
    Returns:
        mnistpts (List[array]): List of point cloud.
        labels (array): class labels. 
    """
    dump_file_dir = osp.join(dump_file_dir, mode)
    dump_file_pts = osp.join(dump_file_dir, "pts.pkl")
    dump_file_labels = osp.join(dump_file_dir, "label.pkl")

    with open(dump_file_pts, "rb") as f:
        mnistpts = pickle.load(f)
    with open(dump_file_labels, "rb") as f:
        labels = pickle.load(f)
    return mnistpts, labels


if __name__ == "__main__":
    data_mnist_dir =  "data" 
    data_mnistpts_dir =  "data_dump"
    # Downloading and preprocessing MNIST dataset
    for mode in ["train", "test"]:
        print(f"Processing {mode} set")
        dump_mnistpts(data_mnist_dir, data_mnistpts_dir, mode)
    print("Done.")
