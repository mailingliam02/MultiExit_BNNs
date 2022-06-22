import models
import to_train
from evaluate import evaluate
from hyperparameters import get_hyperparameters
import datasets

# Specify Hyperparameters (maybe add command line compatibility?)
hyperparameters = get_hyperparameters()

train_loader, val_loader, test_loader = datasets.get_dataloader(hyperparameters["loaders"])
# Evaluate the Network on Test
test_loss_fn = to_train.get_loss_function(hyperparameters["test_loss"])

# Load the Network
print("Getting Network")
# List of models to test
model_list = ["45","60","61","62","63"]
for model_state in model_list:
    path = "/vol/bitbucket/lrc121/ind_proj/MultiExit_BNNs/snapshots/"
    model_type = "best_val_model_"
    print("Testing ", model_type+model_state)
    model_state = path+model_type+model_state
    hyperparameters["network"]["load_model"] = model_state
    # Follow og for info on how to parralelize!
    model = models.get_network(hyperparameters["network"])

    # loss_fn, test_iter, model, gpu
    results = evaluate(test_loss_fn, test_loader,model,hyperparameters["gpu"], 0, hyperparameters["mc_dropout_passes"], create_log = False)

