import to_train.loss.msdnet_loss as loss
from to_train.loss.loss_utils import dict_drop

# Will need to add some functionality if it is not one of the msdnet_losses!
def get_loss_function(hyperparameters):
    loss_f = getattr(loss, hyperparameters['call'])(**dict_drop(hyperparameters, 'call'))
    return loss_f

