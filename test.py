from models.msdnet.msdnet_base import MsdNet
import datasets
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

if __name__ == "__main__":
    hyperparams = dict(          # MSDNet architecture parameters
            call = 'MsdNet',
            in_shape = 32,
            out_dim = 10,
            n_scales = 3,
            n_exits = 11,
            nlayers_to_exit = 4,
            nlayers_between_exits = 2,
            nplanes_mulv = [6, 12, 24],
            nplanes_addh = 1,
            nplanes_init = 1,
            prune = 'min',
            plane_reduction = 0.5,
            exit_width = 128,
            btneck_widths = [4, 4, 4],
            execute_exits = [0,1,2,3], # Determines which exits to return
            )

    model = get_network(hyperparams)
    hyperparameters = dict(dataset_name = "cifar10",
    batch_size = (1,1,1), #(train, val, test)
    augment = True,
    val_split = 0.2,
    )
    train_loader, val_loader, test_loader = datasets.get_dataloader(hyperparameters)
    x,y = next(iter(val_loader))
    output = model(x)
    # print(len(output))
    # print(output[0])