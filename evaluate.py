import torch
from to_train.loss import dict_drop
from to_train.train_utils import validate_model, tab_str, get_device

def evaluate(loss_fn, test_iter, model, gpu, experiment_id):
    # Is this needed again?
    net = model.to(get_device(gpu))
    # Setting train mode to false
    model.eval()
    loss_metrics = loss_fn.metric_names
    val_metrics = validate_model(loss_fn, model, test_iter, gpu)
    print(tab_str(*val_metrics))
    # Log
    log(loss_metrics,val_metrics, experiment_id)
    return val_metrics

def log(loss_metrics,val_metrics, experiment_id):
    float_types = (float, torch.FloatTensor, torch.cuda.FloatTensor)
    value_list = [f'{a:>8.4f}' if isinstance(a, float_types) else f'{a}'
               for a in val_metrics]
    # Store as Pandas csv file?
    joined = list(zip(loss_metrics,value_list))
    with open("log_"+str(experiment_id)+".txt", "w") as file1:
        # Writing data to a file
        file1.write(str(joined))
