import to_train.loss.msdnet_loss as loss
import to_train.loss.resnet_loss as resnet_loss
from to_train.loss.loss_utils import dict_drop

# Inspired from how load classes: https://github.com/mary-phuong/multiexit-distillation/blob/master/main.py 
def get_loss_function(hyperparameters):
    if hyperparameters['call'] == 'ExitEnsembleDistillation':
        loss_f = getattr(resnet_loss, hyperparameters['call'])(**dict_drop(hyperparameters, 'call'))
    else:
        loss_f = getattr(loss, hyperparameters['call'])(**dict_drop(hyperparameters, 'call'))
    return loss_f

