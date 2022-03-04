""" Image directory assumption

    root-image-directory
        --train
            - label1_dir
                - photo1.jpg
                - photo2.jpg
                ...
                - photon.jpg
            - label2_dir
            ...
            - labeln_dir
        --test
            - label1_dir
            - label2_dir
            ...
            - labeln_dir

    So the division of datasets are on level 1 (root is level 0), and the labels of the
    files are on level2
"""

# System packages
import os
from typing import List

# Data Wrangling packages
import pandas as pd

# Pytorch pacakges
from torchvision.io import read_image
from torch.utils.data import Dataset
from msilib.schema import Error

class CustomImageDataset(Dataset):
    def __init__(self, img_dir:str, transform=None, target_transform=None):
        self.img_labels = None
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self._get_labels(img_dir)

    def _get_labels(self, img_dir:str)->List[str]:
        if(self._check_img_dir(img_dir)):
            pass
        

    def _check_img_dir(self, img_dir:str)->bool:
        level1_dir = os.listdir(img_dir)
        
        # check that test, train, validation in. 
        has_test = 'test' in level1_dir
        has_train = 'train' in level1_dir
        
        if(has_test==False or has_train==False):
            raise Error(f"Train({has_train}) and test({has_test}) are not included in the root image directory")
        
        test_is_dir = os.path.isdir(img_dir+'/test')
        train_is_dir = os.path.isdir(img_dir+'/train')

        if(test_is_dir==False or train_is_dir==False):
            raise Error(f"Train({train_is_dir}) and test({test_is_dir}) are not directories")
            
        return True

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label