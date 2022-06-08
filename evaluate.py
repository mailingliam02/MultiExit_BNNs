import torch
from to_train.loss import dict_drop
from to_train.train_utils import validate_model, tab_str, get_device

def evaluate(loss_fn, test_iter, model, gpu):
    # Is this needed again?
    net = model.to(get_device(gpu))
    loss_metrics_string = '\t'.join(loss_fn.metric_names)
    val_metrics = validate_model(loss_fn, model, test_iter, gpu)
    print(tab_str(*val_metrics))
    # Log
    log(loss_metrics_string,val_metrics)

def log(loss_metrics,val_metrics):
    float_types = (float, torch.FloatTensor, torch.cuda.FloatTensor)
    value_strings = (f'{a:>8.4f}' if isinstance(a, float_types) else f'{a}'
               for a in val_metrics)
    with open("log.txt", "w") as file1:
        # Writing data to a file
        file1.write(loss_metrics)
        file1.write(value_strings)
