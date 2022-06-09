import sacred
import torchvision.transforms
import models
import to_train
import datasets
from hyperparameters import get_hyperparameters
from to_train import train_loop
from evaluate import evaluate
from utils import RUNS_DB_DIR

sacred.SETTINGS['CAPTURE_MODE'] = 'sys'
sacred.SETTINGS['HOST_INFO']['INCLUDE_GPU_INFO'] = False
ex = sacred.Experiment()
ex.observers.append(sacred.observers.FileStorageObserver.create(str(RUNS_DB_DIR)))


# Specify Hyperparameters (maybe add command line compatibility?)
hyperparameters = get_hyperparameters()
# Load any Utilities (like a timer, logging...)
ex.add_config(hyperparameters)

@ex.main
def main(_config):
    # Load the dataset
    print("Loading Datasets")
    # Convert hyperparameters so they load into this directly (and the above)!
    # Need to return validation stuff as well!
    train_loader, val_loader, test_loader = datasets.get_dataloader(hyperparameters["loaders"])

    # Load the Network
    print("Creating Network")
    # Follow og for info on how to parralelize!
    model = models.get_network(hyperparameters["network"])

    # Train the Network
    print("Starting Training")
    # Get loss function (if class need to initialize it then run it)
    loss_fn = to_train.get_loss_function(hyperparameters["loss"])
    # Get Optimizer
    optimizer = to_train.get_optimizer(model,hyperparameters["optimizer"])
    # Get Scheduler
    scheduler = to_train.get_scheduler(optimizer,hyperparameters["scheduler"])
    # Train Loop (do i need to return model?)
    model = train_loop(model,optimizer,scheduler,(train_loader,val_loader),loss_fn, gpu = hyperparameters["gpu"], epochs = hyperparameters["n_epochs"]) # model, optimizer, scheduler,  data_loaders, loss_fn, epochs=1, gpu = -1

    # Evaluate the Network on Test
    test_loss_fn = to_train.get_loss_function(hyperparameters["test_loss"])
    # loss_fn, test_iter, model, gpu
    results = evaluate(test_loss_fn, test_loader,model,hyperparameters["gpu"])
    # Save Model

    return results

if __name__ == "__main__":
    ex.run()