from to_train.loss import dict_drop
from to_train.train_utils import validate_model, tab_str, get_device

def evaluate(loss_fn, test_iter, model, gpu):
    # Is this needed again?
    net = model.to(get_device(gpu))
    print('\t'.join(loss_fn.metric_names))
    val_metrics = validate_model(loss_fn, model, test_iter, gpu)
    print(tab_str(*val_metrics))