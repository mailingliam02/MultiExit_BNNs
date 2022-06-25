import argparse
import datasets
from fvcore.nn import FlopCountAnalysis
from model_tester import load_model
from hyperparameters import get_hyperparameters

parser = argparse.ArgumentParser(description="Adding dropout")
parser.add_argument('--dropout_exit', type=bool, default=False)
parser.add_argument('--dropout_p', type=float, default=0.5)
parser.add_argument('--dropout_type', type=str, default=None)
parser.add_argument('--n_epochs', type=int, default=300)
parser.add_argument('--patience', type=int, default=50)
args = parser.parse_args()
# Specify Hyperparameters (maybe add command line compatibility?)
hyperparameters = get_hyperparameters(args)

model_num = "75"
model = load_model(hyperparameters, model_num, model_type = "val")

hyperparameters["loaders"]["batch_size"] = (1,1,1)
train_loader, val_loader, test_loader = datasets.get_dataloader(hyperparameters["loaders"])
inputs = next(iter(train_loader))
flops = FlopCountAnalysis(model,inputs)
print(flops.total())