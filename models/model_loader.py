import torch
from models.msdnet import MsdNet
from models.resnet18 import get_res_net_20, get_res_net_18
from models.vgg import get_vgg_19
from models.vgg import get_vgg_11
def dict_drop(dic, *keys):
    new_dic = dic.copy()
    for key in keys:
        if key in new_dic:
            del new_dic[key]
    return new_dic

# Inspired from how https://github.com/mary-phuong/multiexit-distillation/blob/master/main.py 
# load their networks
def get_network(network_hyperparams):
    if network_hyperparams["load_model"] is not None:
        if torch.cuda.is_available():
            try:
                device = network_hyperparams["gpu_device"]
                model = torch.load(network_hyperparams["load_model"], map_location = device)
            except KeyError:
                model = torch.load(network_hyperparams["load_model"])
        else:
            model = torch.load(network_hyperparams["load_model"], map_location=torch.device('cpu'))
    elif network_hyperparams["call"] == "MsdNet":
        model = MsdNet(**dict_drop(network_hyperparams, "call", "load_model"))
    elif network_hyperparams["call"] == "ResNet18":
        model = get_res_net_18(network_hyperparams["resnet_type"], dict_drop(network_hyperparams, "call", "load_model","resnet_type"))
    elif network_hyperparams["call"] == "ResNet20":
        model = get_res_net_20(network_hyperparams["resnet_type"], dict_drop(network_hyperparams, "call", "load_model","resnet_type"))
    elif network_hyperparams["call"] == "VGG19":
        model = get_vgg_19(network_hyperparams["resnet_type"], dict_drop(network_hyperparams, "call", "load_model","resnet_type"))    
    elif network_hyperparams["call"] == "VGG11":
        model = get_vgg_11(network_hyperparams["resnet_type"], dict_drop(network_hyperparams, "call", "load_model","resnet_type"))
    else:
        raise AttributeError
    return model