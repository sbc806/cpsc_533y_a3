import ml_collections

def get_config():
    """Get the hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Hyperparameters for dataset. 
    config.ratio_tr_data = 0.8 
    config.batch_size = 32 
    config.num_workers = 8 
    config.data_mnistpts_dir = "data_dump"
    config.num_pts = 64 
    config.indim = 2 # the spatial dimension of point clouds. 
    config.order_pts = False 
    config.num_classes = 10  

    # Hyperparameters for models.
    config.model = "PointNet"
    config.outc_list = [128, 128] # number of channels of intermediate FC layers.
    config.downratio = 0.5 # the ratio of number of points in downsampled cloud to the input cloud. 
    config.k = 4 # neighborhood size. 
    config.cn_opt = 'acn' # Context normalization option.

    # Hyperparameters for training.
    config.log_dir = "logs"
    config.use_cuda = True
    config.batch_size = 32
    config.num_epochs = 16
    config.lr = 1e-3 
 
    return config