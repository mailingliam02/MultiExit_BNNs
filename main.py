import sacred
import argparse
import torch
import torchvision.transforms
import models
import to_train
import datasets
from hyperparameters import get_hyperparameters
from to_train import train_loop
from evaluate import evaluate
from utils import RUNS_DB_DIR
from model_tester import FullAnalysis

sacred.SETTINGS['CAPTURE_MODE'] = 'sys'
sacred.SETTINGS['HOST_INFO']['INCLUDE_GPU_INFO'] = False
ex = sacred.Experiment()
ex.observers.append(sacred.observers.FileStorageObserver.create(str(RUNS_DB_DIR)))

parser = argparse.ArgumentParser(description="Adding dropout")
parser.add_argument('--dropout_exit', type=bool, default=False)
parser.add_argument('--dropout_p', type=float, default=0.5)
parser.add_argument('--dropout_type', type=str, default=None)
parser.add_argument('--n_epochs', type=int, default=300)
parser.add_argument('--patience', type=int, default=300)
parser.add_argument('--full_analysis_and_save', type=bool, default = False)
parser.add_argument('--single_exit', type=bool, default=False)
parser.add_argument('--backbone', type=str, default = "msdnet")
parser.add_argument('--grad_clipping', type=float, default = 2)
parser.add_argument('--gpu', type=int,default=0)
parser.add_argument('--val_split', type=float, default = 0.1)
parser.add_argument('--reducelr_on_plateau', type=bool, default = False)
args = parser.parse_args()
#https://stackoverflow.com/questions/44561722/why-in-argparse-a-true-is-always-true

# Specify Hyperparameters (maybe add command line compatibility?)
hyperparameters = get_hyperparameters(args)
# Load any Utilities (like a timer, logging...)
ex.add_config(hyperparameters)

@ex.main
def main(_config):
    # Get experiment ID for logging purposes
    experiment_id = ex.current_run._id
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
    model = train_loop(model,optimizer,scheduler,(train_loader,val_loader),
            loss_fn, experiment_id, gpu = hyperparameters["gpu"], 
            epochs = hyperparameters["n_epochs"], 
            patience = hyperparameters["patience"],
            max_norm = hyperparameters["max_norm"],
            val_loss_type = hyperparameters["val_loss_type"]) # model, optimizer, scheduler,  data_loaders, loss_fn, epochs=1, gpu = -1

    # Evaluate the Network on Test
    test_loss_fn = to_train.get_loss_function(hyperparameters["test_loss"])
    # loss_fn, test_iter, model, gpu
    results = evaluate(test_loss_fn, test_loader,model,hyperparameters["gpu"], experiment_id, hyperparameters["mc_dropout_passes"])
    # Save Model
    torch.save(model, "./MultiExit_BNNs/snapshots/final_model_"+str(experiment_id))
    if args.full_analysis_and_save:
        dropout = False
        if args.dropout_exit or args.dropout_type is not None:
            dropout = True
        full_analyzer = FullAnalysis(model, test_loader, gpu = hyperparameters["gpu"], 
            mc_dropout = dropout, mc_passes = hyperparameters["mc_dropout_passes"])
        full_analyzer.all_experiments(experiment_id)
        full_analyzer.save_validation(experiment_id, val_loader)
    return results

if __name__ == "__main__":
    ex.run()