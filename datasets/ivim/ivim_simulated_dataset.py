import os
import torch
from torch.utils.data import Dataset
import pickle
import math
import numpy as np
from datasets.ivim.simulations import sim_signal


class SimulatedIVIMDataset(Dataset):
    def __init__(self, img_dir, SNR = 5, dataset_type = "train", 
                transform=None, target_transform=None, valid_split = 0.2):
        self.snr = SNR
        # Random seed
        random_state = 42
        # Generate data if not already existing
        self.file_dir = os.path.join(img_dir, "simulated_data.npy")
        self.b_values = self._get_bvalues()
        if not os.path.exists(self.file_dir):
            data_sim, D, f, Dp = sim_signal(self.snr, self.b_values)
            with open(self.file_dir, 'wb') as file:
                np.save(file, data_sim)
                np.save(file, D)
                np.save(file, f)
                np.save(file, Dp)
        else:
            with open(self.file_dir, 'rb') as file:
                data_sim = np.load(file)
                D = np.load(file)
                f = np.load(file)
                Dp = np.load(file)
                # What to do with label?
        train_split = 1-valid_split*2
        train_vals = math.floor(train_split*data_sim.shape[0])
        val_vals = math.floor(valid_split*data_sim.shape[0])+train_vals
        if dataset_type == "train":
            self.data = data_sim[:train_vals,:]
            self.D = D[:train_vals]
            self.f = f[:train_vals]
            self.Dp = Dp[:train_vals]

        elif dataset_type == "val":
            self.data = data_sim[train_vals:val_vals,:]
            self.D = D[train_vals:val_vals]
            self.f = f[train_vals:val_vals]
            self.Dp = Dp[train_vals:val_vals]          

        elif dataset_type == "test":
            self.data = data_sim[val_vals:,:]
            self.D = D[val_vals:]
            self.f = f[val_vals:]
            self.Dp = Dp[val_vals:]  
        else:
            raise AttributeError
        self.transform = transform
        self.target_transform = target_transform
        #print(self.data.shape, self.D.shape, self.f.shape, self.Dp.shape)

    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        # Get data
        ivim_signal = self.data[idx,:]
        # Extract label
        label = (torch.tensor(self.D[idx][0]),torch.tensor(self.f[idx][0]),torch.tensor(self.Dp[idx][0]))
        # Apply transformations
        if self.transform:
            ivim_signal = self.transform(ivim_signal)
        if self.target_transform:
            label = self.target_transform(label)
        return torch.from_numpy(ivim_signal), label


    def _get_bvalues(dataset):
        return [0, 0, 0, 0, 0, 0, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80,80, 80, 80, 80, 80, 80, 80, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 500, 500,500,500, 500,500, 500, 500, 500]

if __name__ == "__main__":
    from simulations import sim_signal
    dataset = SimulatedIVIMDataset("/mnt/c/PythonScripts/ind_proj/github_repo/MultiExit_BNNs/data/ivim_simulated")
    print(len(dataset))
    print(dataset[5])