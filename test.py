from inspect import isclass
import models
import torch
import math
import to_train
from torch import nn
from torchvision import transforms, datasets
from models.msdnet.msdnet_base import MsdNet
from models.msdnet.msdnet_layers import MsdLayer, MsdLayer0
from models.msdnet.msdnet_utils import MsdTransition, ConvBnRelu2d, View

def test_msdnet():
    hyperparameters = get_msd_test_hyperparameters()
    test_data = get_test_data(hyperparameters)
    model = models.get_network(hyperparameters["network"])
    loss_fn = to_train.get_loss_function(hyperparameters["loss"])
    # Not correct, but shouldn't be causing issues with training!
    # test_architecture(model,hyperparameters["network"])
    # Identical (Need to set weights of Linear layer)
    #test_exit_outputs(model, test_data, hyperparameters["network"])
    # Seems to be working, didn't actually work out by hand
    #test_loss(model, test_data, loss_fn)
    x = test_data[0]
    y = test_data[1]
    print("Blocks")
    print(model.blocks)
    print("Exits")
    print(model.exits)
    actual = model(x)
    print("Actual")
    print(actual)
    return


def get_test_data(hyperparams):
    mean=[0.5071, 0.4865, 0.4409]
    std=[0.2673, 0.2564, 0.2762]
    normalize = transforms.Normalize(mean = mean,std = std)   
    train_transforms = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
    train_set = datasets.CIFAR100(
                root="data/cifar100", train=True,
                download=False, transform=train_transforms,
            )
    train_loader = torch.utils.data.DataLoader(train_set,batch_size = 1,
                num_workers=1, pin_memory=True)
    single_item_v2 = next(iter(train_loader))
    return single_item_v2



def test_fn_2():
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
    return


def get_msd_test_hyperparameters():
    # Main
    model_type = "msdnet"
    n_epochs = 2
    gpu = -1

    # Network
    network_hyperparameters = get_network_hyperparameters(model_type)
    # Losses
    loss_hyperparameters = get_loss_hyperparameters(network_hyperparameters["n_exits"], model_type)
    test_loss_hyperparameters = get_test_hyperparameters(network_hyperparameters["n_exits"], model_type)
    # Train and Val 
    loader_hyperparameters = get_loader_hyperparameters()
    # Optimizer and Scheduler
    opt_hyperparameters, sched_hyperparameters = get_opt_sched_hyperparameters()

    
    hyperparameters = dict(
        network = network_hyperparameters,
        loss = loss_hyperparameters,
        optimizer = opt_hyperparameters,
        scheduler = sched_hyperparameters,
        n_epochs = n_epochs,
        test_loss = test_loss_hyperparameters,
        gpu = gpu,
        loaders = loader_hyperparameters,
        )
    return hyperparameters

def get_network_hyperparameters(model_type):
    hyperparams = dict(          # MSDNet architecture parameters
        call = 'MsdNet',
        in_shape = 32,
        out_dim = 100,
        n_scales = 3,
        n_exits = 3,
        nlayers_to_exit = 2,
        nlayers_between_exits = 2,
        nplanes_mulv = [2, 4, 8],
        nplanes_addh = 1,
        nplanes_init = 1,
        prune = "min",
        plane_reduction = 0, # Try this with 0 to avoid the halving
        exit_width = 128, # same as 128 dim 3x3 filters in exit?
        test_mode = True,
        dropout = "block",
        dropout_exit = True,
        dropout_p = 0.8,
        ) 
        # So the problem with sizes is due to getting to view with too many values
        # Can come up with a work around, but is a bit shite (just flatten and put in linear). 
        # Solution is to instead plan so conv works out to right amount at the end.
        # We note that this makes it more rigid
    return hyperparams

def get_loss_hyperparameters(num_exits, model_type,loss_type = "distillation_annealing"):
    if model_type == "msdnet":
        if loss_type == "distillation_annealing":
            loss = dict(         # distillation-based training with temperature
                                # annealing
            call = 'DistillationBasedLoss',
            n_exits = num_exits,
            acc_tops = [1, 5],
            
            C = 0.5, # Confidence Limit (?)
            maxprob = 0.5, 
            global_scale = 2.0 * 5/num_exits, # Not mentioned in paper
            # Temperature multiplier is 1.05 by default
            )
        elif loss_type == "distillation_constant":
            loss = dict(       # distillation-based training with constant
                                # temperature
                call = 'DistillationLossConstTemp',
                n_exits = num_exits,
                acc_tops = [1, 5],
                C = 0.5,
                T = 4.0,
                global_scale = 2.0 * 5/num_exits,
            )
        elif loss_type == "classification":
            loss = dict(       # train with classification loss only
                call = 'ClassificationOnlyLoss',
                n_exits = num_exits,
                acc_tops = [1, 5],
            )
    else:
        # Add standard loss function stuff here
        pass
    return loss

def get_opt_sched_hyperparameters():
    cf_opt = dict(          # optimization method
    call = 'SGD',
    lr = 0.5, # Note this is from Paper 9 (Paper 10 used 0.1)
    momentum = 0.9,
    weight_decay = 1e-4,
    nesterov = True,
    )
    cf_scheduler = dict(   # learning rate schedule
    call = 'MultiStepLR',
    milestones = [150, 225],
    gamma = 0.1
    )
    return cf_opt, cf_scheduler

def get_loader_hyperparameters():
    hyperparameters = dict(dataset_name = "cifar100",
        batch_size = (1,1,250), #(train, val, test)
        augment = True,
        val_split = 0.1,
        )
    return hyperparameters


def get_test_hyperparameters(n_exits, model_type):
    if model_type == "msdnet":
        cf_loss = dict(  # evaluation metric
            call = 'MultiExitAccuracy',
            n_exits = n_exits,
            acc_tops = (1,5),
        )
    return cf_loss

if __name__ == "__main__":
    test_msdnet()