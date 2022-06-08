

def get_hyperparameters():
    # Main
    model_type = "msdnet"
    n_epochs = 1
    gpu = 0

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
    if model_type == "msdnet":
        hyperparams = dict(          # MSDNet architecture parameters
            call = 'MsdNet',
            in_shape = 32,
            out_dim = 100,
            n_scales = 3,
            n_exits = 11,
            nlayers_to_exit = 4,
            nlayers_between_exits = 2,
            nplanes_mulv = [6, 12, 24],
            nplanes_addh = 1,
            nplanes_init = 1,
            prune = 'min',
            plane_reduction = 0.5, # Try this with 0 to avoid the halving
            exit_width = 128, # same as 128 dim 3x3 filters in exit?
            )
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
        batch_size = (64,250,250), #(train, val, test)
        augment = True,
        val_split = 0.1,
        )
    return hyperparameters


def get_test_hyperparameters(n_exits, model_type):
    if model_type == "msdnet":
        cf_loss = dict(  # evaluation metric
            call = 'MultiExitAccuracy',
            n_exits = n_exits,
            acc_tops = (5),
        )
    return cf_loss
