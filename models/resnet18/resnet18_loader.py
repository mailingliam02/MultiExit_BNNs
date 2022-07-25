from utils import dict_drop
from models.resnet18.resnet18 import ResNet18EarlyExitLee, ResNet18Base, ResNet18MC, ResNet18MCEarlyExitLee

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
        # n_exits, out_dim
        return ResNet18Base(**dict_drop(network_hyperparams,"call", "load_model",
            "resnet_type","dropout","dropout_exit", "dropout_p"))

    elif ensemble == "early_exit_lee":
        # n_exits, out_dim
        return ResNet18EarlyExitLee(**dict_drop(network_hyperparams,"call", "load_model",
            "resnet_type","dropout","dropout_exit", "dropout_p"))

    elif ensemble == "mc":
        # n_exits, out_dim, dropout, dropout_exit, dropout_p
        return ResNet18MC(**dict_drop(network_hyperparams,"call", "load_model",
            "resnet_type"))
     
    elif ensemble == "mc_early_exit":
        # n_exits, out_dim, dropout, dropout_exit, dropout_p
        return ResNet18MCEarlyExitLee(**dict_drop(network_hyperparams,"call", "load_model",
            "resnet_type"))