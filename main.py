import torch
import torchvision
import torchvision.transforms
import models
import to_train
import datasets
from hyperparameters import get_hyperparameters
from to_train import train_loop
from evaluate import evaluate

# Specify Hyperparameters (maybe add command line compatibility?)
hyperparameters = get_hyperparameters()
# Load any Utilities (like a timer, logging...)


# Load the dataset
print("Loading Datasets")
transforms = datasets.get_transforms()
# Convert hyperparameters so they load into this directly (and the above)!
# Need to return validation stuff as well!
train_loader, test_loader = datasets.get_dataloader("cifar10", 64, transforms, download = False)

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
model = train_loop(model,optimizer,scheduler,(train_loader,test_loader),loss_fn, gpu = hyperparameters["gpu"]) # model, optimizer, scheduler,  data_loaders, loss_fn, epochs=1, gpu = -1

# Evaluate the Network on Test
test_loss_fn = to_train.get_loss_function(hyperparameters["test_loss"])
evaluate(hyperparameters["test"], test_loss_fn, test_loader,model,hyperparameters["gpu"])

# Access predict via to_train.predict
# Need to define acc or f1 somewhere

# Save Results


# Save Model