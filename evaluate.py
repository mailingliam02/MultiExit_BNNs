import torch
import numpy as np
from to_train.loss import dict_drop
from to_train.train_utils import validate_model_acc, tab_str, get_device

def evaluate(loss_fn, test_iter, model, gpu, experiment_id, mc_dropout_passes):
    # Is this needed again?
    net = model.to(get_device(gpu))
    # Setting train mode to false
    model.eval()
    loss_metrics = loss_fn.metric_names
    test_metrics = np.zeros((mc_dropout_passes,len(loss_metrics)))
    for i in range(mc_dropout_passes):
        values = validate_model_acc(loss_fn, model, test_iter, gpu)
        values[-1] = values[-1].item()
        test_metrics[i] = values
    averaged_test_metrics = list(np.average(test_metrics, axis = 0))
    print(tab_str(*averaged_test_metrics))
    # Log
    log(loss_metrics,averaged_test_metrics, experiment_id)
    return averaged_test_metrics

def log(loss_metrics,val_metrics, experiment_id):
    float_types = (float, torch.FloatTensor, torch.cuda.FloatTensor)
    value_list = [f'{a:>8.4f}' if isinstance(a, float_types) else f'{a}'
               for a in val_metrics]
    # Store as Pandas csv file?
    joined = list(zip(loss_metrics,value_list))
    with open("log_"+str(experiment_id)+".txt", "w") as file1:
        # Writing data to a file
        file1.write(str(joined))
