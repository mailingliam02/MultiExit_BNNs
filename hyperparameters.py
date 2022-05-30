

def get_hyperparameters():
    # Network
    network_hyperparameters = get_network_hyperparameters()
    # Losses
    loss_hyperparameters = get_loss_hyperparameters(network_hyperparameters["n_exits"])
    test_loss_hyperparameters = get_test_hyperparameters(network_hyperparameters["n_exits"])
    # Train and Val 
    loader_hyperparameters = get_loader_hyperparameters()
    # Optimizer and Scheduler
    opt_hyperparameters, sched_hyperparameters = get_opt_sched_hyperparameters()
    n_epochs = 1
    gpu = 0
    
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

def get_network_hyperparameters():
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
        plane_reduction = 0.5,
        exit_width = 128,
        btneck_widths = [4, 4, 4],
        )
    return hyperparams

def get_loss_hyperparameters(num_exits, loss_type = "distillation_annealing"):
    if loss_type == "distillation_annealing":
        loss = dict(         # distillation-based training with temperature
                            # annealing
        call = 'DistillationBasedLoss',
        n_exits = num_exits,
        acc_tops = [1, 5],
        
        C = 0.5,
        maxprob = 0.5,
        global_scale = 2.0 * 5/num_exits,
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
    return loss

def get_opt_sched_hyperparameters():
    cf_opt = dict(          # optimization method
    call = 'SGD',
    lr = 0.1,
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
    hyperparameters = dict(dataset_name = "cifar10",
        batch_size = (64,250,250), #(train, val, test)
        augment = True,
        val_split = 0.2,
        )
    return hyperparameters


def get_test_hyperparameters(n_exits):
    cf_loss = dict(  # evaluation metric
        call = 'MultiExitAccuracy',
        n_exits = n_exits,
        acc_tops = (1, 5),
    )
    return cf_loss
