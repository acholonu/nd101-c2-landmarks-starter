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
from typing import Tuple

# Pytorch pacakges
from torchvision.io import read_image
from torch.utils.data import Dataset

"""So I need to flatten the dataset.  for labels->dictionary key=image id, value = label, 
train partition = all image ids, but they should follow the label order.
"""

# Reference: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class CustomImageDataset(Dataset):
    def __init__(self, img_dir:str, transform=None, target_transform=None,  folders:str=['train','test']):
        self.img_labels = None
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.folders = folders
        # key is the folder name from self.folders, starting indices of images in that folder
        self.folder_indicies = {}  

        self._generate_img_indices(img_dir)
        
    def _generate_img_indices(self,img_dir:str):
        if(self._check_img_dir(img_dir)):
            images = []
            labels = {}
            index = 0
            for folder in self.folders:
                root_dir = os.path.join(img_dir,folder).replace(" ","\\ ") # Mac/Linus specific
                start_index = index
                
                # Image ids are the path to the image.
                for root, dirs, files in os.walk(root_dir, topdown=False):
                    for dir_name in dirs:    
                        for file_name in files:
                            path=(os.path.join(root, dir_name, file_name.replace(" ","\\ "))) #id
                            class_label = dir_name
                            labels[path] = class_label
                            images.append(path)
                            index = index + 1
                self.folder_indicies[folder] = (start_index, index)

            self.images = images
            self.img_labels = labels
        else:
            raise Exception("Unable to locate images and labels")
        return images, labels

      
    def check_img_dir(self, img_dir:str)->bool:
        level1_dir = os.listdir(img_dir)
        
        # check that test, train, validation in. 
        for folder in self.folders:
            has_folder = (
                (folder in level1_dir) and 
                (os.path.isdir(os.path.join(img_dir,folder)))
            )
            print(f"{folder} is a sub directory in the image directory?: {has_folder}")
            if has_folder == False:
                return(False)

        return(True)


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images.iloc[idx])
        print(f"image path: {img_path}")

        image = read_image(img_path)
        label = self.img_labels.iloc[idx]
        print(f"label: {label}")

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def get_folder_img_index_range(self, folder:str)->Tuple:
        return(self.folder_indicies[folder])