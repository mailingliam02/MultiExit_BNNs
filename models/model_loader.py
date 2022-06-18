import torch
from models.msdnet import MsdNet
from models.ivimnet import IVIMNet

def dict_drop(dic, *keys):
    new_dic = dic.copy()
    for key in keys:
        if key in new_dic:
            del new_dic[key]
    return new_dic



def get_network(network_hyperparams):
    if network_hyperparams["load_model"] is not None:
        model = torch.load(network_hyperparams["load_model"])
    elif network_hyperparams["call"] == "MsdNet":
        model = MsdNet(**dict_drop(network_hyperparams, "call", "load_model"))
    elif network_hyperparams["call"] == "IVIMNet":
        model = IVIMNet(**dict_drop(network_hyperparams, "call", "load_model"))
    else:
        raise AttributeError
    return model