from models.msdnet import MsdNet

def dict_drop(dic, *keys):
    new_dic = dic.copy()
    for key in keys:
        if key in new_dic:
            del new_dic[key]
    return new_dic



def get_network(network_hyperparams):
    if network_hyperparams["call"] == "MsdNet":
        model = MsdNet(**dict_drop(network_hyperparams, "call"))
    else:
        raise AttributeError
    return model