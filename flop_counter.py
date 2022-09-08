import argparse
import datasets
from fvcore.nn import FlopCountAnalysis
from model_tester import ResourceLoader
from hyperparameters import get_hyperparameters

parser = argparse.ArgumentParser(description="Adding dropout")
parser.add_argument('--dropout_exit', type=bool, default=False)
parser.add_argument('--dropout_p', type=float, default=0.5)
parser.add_argument('--dropout_type', type=str, default=None)
parser.add_argument('--n_epochs', type=int, default=300)
parser.add_argument('--patience', type=int, default=50)
args = parser.parse_args()
# Specify Hyperparameters (maybe add command line compatibility?)
model_num = "67"
loader = ResourceLoader()
model = loader.get_model(model_num, model_type = "val")
print(model.modules)
loader_hyperparameters = dict(dataset_name = "cifar100",
    batch_size = (1,1,1), #(train, val, test) 
    # train and val batch sizes should be the same for plotting purposes
    augment = True,
    val_split = 0.1,
    )
train_loader, val_loader, test_loader = datasets.get_dataloader(loader_hyperparameters)
inputs = next(iter(train_loader))

flops = FlopCountAnalysis(model,inputs[0])
print(flops.total())
print(flops.by_module())