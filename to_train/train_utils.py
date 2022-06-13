import torch
from to_train.loss import dict_drop

def get_device(gpu):
    return torch.device(f'cuda:{gpu}' if gpu >= 0 else 'cpu')

def predict(model, X, gpu):
    device = get_device(gpu)
    X = X.to(device)
    model.to(device)
    scores = model(X)
    # Get last score
    _, pred = scores[-1].max(1)
    return pred.cpu().numpy()

def get_optimizer(model, hyperparameters):
    Opt = getattr(torch.optim, hyperparameters['call'])
    opt = Opt(model.parameters(), **dict_drop(hyperparameters, 'call'))
    return opt

def get_scheduler(opt, hyperparameters):
    Scheduler = getattr(torch.optim.lr_scheduler, hyperparameters['call'])
    scheduler = Scheduler(opt, last_epoch=-1,
                          **dict_drop(hyperparameters, 'call'))
    return scheduler

# Needs to be rewritten
def validate_model(loss_f, net, val_iter, gpu):
    metrics = []
    for val_tuple in val_iter:
        val_tuple = [t.to(get_device(gpu)) for t in val_tuple]
        metrics += [loss_f.metrics(net, *val_tuple)]
    return [sum(metric) / len(metric) for metric in zip(*metrics)]

def tab_str(*args):
    float_types = (float, torch.FloatTensor, torch.cuda.FloatTensor)
    strings = (f'{a:>8.4f}' if isinstance(a, float_types) else f'{a}'
               for a in args)
    return '\t'.join(strings)