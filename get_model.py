from models.fcnet import FcNet
from models.pointnet import PointNet
from models.pointnet2 import PointNet2
from models.pointconv import PointConv 
from models.acne import Acne 


def get_model(config):
    """Get a model according configuration."""
    if config.model == "fcnet": 
        model = FcNet(config) # We did this in the assignment1.
    elif config.model == "pointnet":
        model = PointNet(config)
    elif config.model == "pointnet2":
        model = PointNet2(config)
    elif config.model == "pointconv":
        model = PointConv(config)
    elif config.model == "acne":
        assert config.cn_opt == 'acn'
        model = Acne(config)
    elif config.model == "cne":
        assert config.cn_opt == 'cn'
        model = Acne(config)
    else:
        raise NotImplementedError 

    # Calculate number of parameters. 
    num_parameters = sum([x.nelement() for x in model.parameters()])
    print(f"The number of parameters in {config.model}: {num_parameters/1000:9.2f}k")
    return model 

