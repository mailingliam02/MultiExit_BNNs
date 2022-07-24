import to_train.loss.msdnet_loss as loss
import to_train.loss.resnet_loss as resnet_loss
from to_train.loss.loss_utils import dict_drop

# Will need to add some functionality if it is not one of the msdnet_losses!
def get_loss_function(hyperparameters):
    if hyperparameters['call'] == 'ExitEnsembleDistillation':
        loss_f = getattr(resnet_loss, hyperparameters['call'])(**dict_drop(hyperparameters, 'call'))
    else:
        loss_f = getattr(loss, hyperparameters['call'])(**dict_drop(hyperparameters, 'call'))
    return loss_f

