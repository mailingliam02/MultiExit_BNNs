from utils import dict_drop
from models.resnet18.resnet18_basic import ResNet18, ResNet18MCDrop
from models.resnet18.resnet18_basic import ResNet18EarlyExitLee

def get_res_net_18(ensemble, network_hyperparams):
    # Network Hyperparams (call = "ResNet18",
            #resnet_type = "early_exit",
            #load_model = None,
            #out_dim = 100,
            #dropout = args.dropout_type,
            #dropout_exit = args.dropout_exit,
            #dropout_p = args.dropout_p,
            #n_exits = 4)
    if ensemble is None:
        # out_dim
        return ResNet18(**dict_drop(network_hyperparams,"call", "load_model",
            "resnet_type","dropout","dropout_exit", "dropout_p","n_exits"))

    elif ensemble == "early_exit_lee":
        # n_exits, out_dim
        return ResNet18EarlyExitLee(**dict_drop(network_hyperparams,"call", "load_model",
            "resnet_type","dropout","dropout_exit", "dropout_p"))