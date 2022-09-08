import os
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image

# Most of this class has been taken from https://github.com/jonahanton/SSL_medicalimaging/blob/main/datasets/custom_chestx_dataset.py
# which itself was adapted from https://github.com/linusericsson/ssl-transfer/tree/main/datasets/chestx.py
# The splits are instead defined by patients rather than randomly dividing the images
class CustomChestXDataset(Dataset):
    def __init__(self, root = "", train = False, transform=None, target_transform=None, download=False):
        """
        Args:
            img_dir (string): path to dataset
        """
        self.root = root
        self.img_path = self.root + "/images/"
        self.csv_path = self.root + "/Data_Entry_2017.csv"
        self.used_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"]

        self.labels_maps = {"Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 2, "Infiltration": 3, "Mass": 4, "Nodule": 5,  "Pneumothorax": 6}

        labels_set = []

        # Transforms
        self.transform = transform
        self.target_transform = target_transform
        # Read the csv file
        df = pd.read_csv(self.csv_path)
        # Split by Patient
        random_state = 42
        # Same as https://arxiv.org/abs/1711.05225
        self.train_len = 24644
        self.val_len = 3081
        self.test_len = 3080
        random.seed(random_state)
        self.all_patients = list(range(self.train_len+self.val_len+self.test_len))
        random.shuffle(self.all_patients)
        self.train_patients = self.all_patients[:self.train_len]
        self.val_patients = self.all_patients[self.train_len:self.train_len+self.val_len]
        self.test_patients = self.all_patients[self.train_len+self.val_len:]
        df.columns = [c.replace(' ', '_') for c in df.columns]
        if train == "train":
            self.data_info = df[pd.DataFrame(df.Patient_ID.tolist()).isin(self.train_patients).any(1).values]
        elif train == "val":
            self.data_info = df[pd.DataFrame(df.Patient_ID.tolist()).isin(self.val_patients).any(1).values]
        else:
            self.data_info = df[pd.DataFrame(df.Patient_ID.tolist()).isin(self.test_patients).any(1).values]

        # First column contains the image paths
        self.image_name_all = np.asarray(self.data_info.iloc[:, 0])
        self.labels_all = np.asarray(self.data_info.iloc[:, 1])

        self.image_name  = []
        self.labels = []


        for name, label in zip(self.image_name_all,self.labels_all):
            label = label.split("|")

            if len(label) == 1 and label[0] != "No Finding" and label[0] != "Pneumonia" and label[0] in self.used_labels:
                self.labels.append(self.labels_maps[label[0]])
                self.image_name.append(name)
    
        self.data_len = len(self.image_name)

        self.image_name = np.asarray(self.image_name)
        self.labels = np.asarray(self.labels)      

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_name[index]
        # Open image
        img_path = os.path.join(self.img_path, single_image_name)
        img = Image.open(img_path).convert('RGB')
        # Transform
        if self.transform:
            img = self.transform(img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.labels[index] # int64
        single_image_label = np.float32(single_image_label) #convert
        if self.target_transform:
            single_image_label = self.target_transform(single_image_label)
        return img, single_image_label

    def __len__(self):
        return self.data_len