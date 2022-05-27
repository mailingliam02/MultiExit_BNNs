import torch
import torchvision
import os

def get_dataloader(dataset_name, batch_size, transforms, shuffle = True, download = False):
    train_set, test_set = get_dataset(dataset_name, transforms, download)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size = batch_size, shuffle = shuffle)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size = batch_size, shuffle = shuffle)
    return train_loader, test_loader 

def get_dataset(dataset_name, transforms, download):
    #print(os.getcwd())  #MultiExit_BNNS/
    if dataset_name == "cifar10":
        # Check the root directory is the same here!
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=download, transform=transforms)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                download=download, transform=transforms)
    return train_set,test_set

