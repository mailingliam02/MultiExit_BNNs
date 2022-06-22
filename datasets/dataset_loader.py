import torch
import torchvision
import os
import torch
import random
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from datasets.ivim import SimulatedIVIMDataset

def get_dataloader(hyperparameters, random_seed = None):
    if random_seed is None:
        random_seed = random.randint(1,1000)
    dataset_loader = DatasetLoader(dataset_name = hyperparameters["dataset_name"], 
        batch_size = hyperparameters["batch_size"], augment = hyperparameters["augment"], 
        random_seed = random_seed, valid_split = hyperparameters["val_split"])
    train_loader, val_loader, test_loader = dataset_loader.get_dataloaders()
    return train_loader, val_loader, test_loader 

# Inspired by https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
class DatasetLoader:
    def __init__(self,dataset_name= "cifar10", batch_size = (64,64,250), augment = False, random_seed = 42, valid_split = 0.2):
        self.dataset_name = dataset_name
        self.train_batch_size = batch_size[0]
        self.val_batch_size = batch_size[1]
        self.test_batch_size = batch_size[2]
        self.augment = augment
        self.valid_size = valid_split
        self.random_seed = random_seed
        self.data_dir = "./data/"+self.dataset_name
        self._get_transforms()
        self._get_dataset()
        if self._sampler_needed():
            self._get_samplers()
        else:
            self.train_sampler = None

    def _sampler_needed(self):
        datasets_without_val = ["cifar10","cifar100","imagenet"]
        if self.dataset_name in datasets_without_val:
            self.shuffle = False
            return True
        else:
            self.shuffle = True
            return False

    def _get_transforms(self):
        if self.dataset_name == "cifar10":
            self.mean=[0.4914, 0.4822, 0.4465]
            self.std=[0.2023, 0.1994, 0.2010]
        elif self.dataset_name == "cifar100":
            self.mean=[0.5071, 0.4865, 0.4409]
            self.std=[0.2673, 0.2564, 0.2762]
        elif self.dataset_name == "ivim_simulated":
            # Transforms are applied in class (as is not image)
            self.train_transforms = None
            self.val_transforms = None
            self.test_transforms = None
            return None
        
        normalize = transforms.Normalize(mean = self.mean,std = self.std)   
        self.val_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
        if self.augment:
            self.train_transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.train_transforms = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        return None

    def _get_dataset(self):
        #print(os.getcwd())  #MultiExit_BNNS/
        if self.dataset_name == "cifar10":
            self.train_set = datasets.CIFAR10(
                root=self.data_dir, train=True,
                download=False, transform=self.train_transforms,
            )
            self.val_set = datasets.CIFAR10(
                root=self.data_dir, train=True,
                download=False, transform=self.val_transforms,
            )
            self.test_set = datasets.CIFAR10(
                root=self.data_dir, train=False,
                download=False, transform=self.test_transforms,
            )
        elif self.dataset_name == "cifar100":
            self.train_set = datasets.CIFAR100(
                root=self.data_dir, train=True,
                download=False, transform=self.train_transforms,
            )
            self.val_set = datasets.CIFAR100(
                root=self.data_dir, train=True,
                download=False, transform=self.val_transforms,
            )
            self.test_set = datasets.CIFAR100(
                root=self.data_dir, train=False,
                download=False, transform=self.test_transforms,
            )
        elif self.dataset_name == "ivim_simulated":
            self.train_set = SimulatedIVIMDataset(self.data_dir, dataset_type = "train",
                transform=self.train_transforms, target_transform=None, valid_split = self.valid_size)
            self.val_set = SimulatedIVIMDataset(self.data_dir, dataset_type = "val",
                transform=self.val_transforms, target_transform=None, valid_split = self.valid_size)
            self.test_set = SimulatedIVIMDataset(self.data_dir, dataset_type = "test",
                transform=self.test_transforms, target_transform=None, valid_split = self.valid_size)
            print(len(self.train_set),len(self.val_set),len(self.test_set))
        elif self.dataset_name == "ivim_actual":
            pass
        return None
    
    def _get_samplers(self):
        num_train = len(self.train_set)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))

        if self.shuffle:
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.valid_sampler = SubsetRandomSampler(valid_idx)
        return None

    def get_dataloaders(self):
        if self.train_sampler is not None:
            train_loader = torch.utils.data.DataLoader(self.train_set,batch_size = self.train_batch_size, sampler=self.train_sampler,
                num_workers=1, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(self.val_set,batch_size = self.val_batch_size, sampler=self.train_sampler,
                num_workers=1, pin_memory=True)
        else:
            train_loader = torch.utils.data.DataLoader(self.train_set,batch_size = self.train_batch_size, shuffle = self.shuffle,
                num_workers=1, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(self.val_set,batch_size = self.val_batch_size, shuffle = self.shuffle,
                num_workers=1, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(self.test_set,batch_size = self.test_batch_size, pin_memory = True)
        return train_loader, val_loader, test_loader

