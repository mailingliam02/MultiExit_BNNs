

def get_hyperparameters():
    # Network
    network_hyperparameters = get_network_hyperparameters()
    # Losses
    loss_hyperparameters = get_loss_hyperparameters(network_hyperparameters["n_exits"])
    # Train and Val 
    train_hyperparameters, val_hyperparameters = get_train_val_hyperparameters()
    # Optimizer and Scheduler
    opt_hyperparameters, sched_hyperparameters = get_opt_sched_hyperparameters()
    test_hyperparameters, test_loss_hyperparameters, test_batch_size = get_test_hyperparameters()
    batch_size = 64
    val_batch_size = 250
    n_epochs = 300
    gpu = 0
    
    hyperparameters = dict(
        network = network_hyperparameters,
        loss = loss_hyperparameters,
        train = train_hyperparameters,
        val = val_hyperparameters,
        optimizer = opt_hyperparameters,
        scheduler = sched_hyperparameters,
        batch_size = batch_size,
        val_batch_size = val_batch_size,
        n_epochs = n_epochs,
        test = test_hyperparameters,
        test_loss = test_loss_hyperparameters,
        test_batch_size = test_batch_size,
        gpu = gpu,
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

def get_train_val_hyperparameters():
    cf_trn = dict(          # training set parameters
    call = 'Cifar100',
    n_per_class = 150,  # number of images per class (including validation)
    nval_per_class = 50,
    augment = True,     # data augmentation
    seed = 0,
    )
    cf_val = cf_trn.copy()
    cf_val['augment'] = False
    return cf_trn, cf_val

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

def get_test_hyperparameters():
    cf_test = dict(  # test dataset
        call = 'Cifar100',
        seed = 0,
    )
    
    cf_loss = dict(  # evaluation metric
        call = 'MultiExitAccuracy',
        n_exits = 11,
        acc_tops = (1, 5),
    )
    batch_size = 250
    return cf_test, cf_loss, batch_size
