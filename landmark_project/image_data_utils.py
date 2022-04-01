"""Image directory assumption

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
import re
import argparse
from matplotlib import transforms
import numpy as np
from typing import Tuple

# Pytorch pacakges
from torchvision.io import read_image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

"""So I need to flatten the dataset.  for labels->dictionary key=image id, value = label, 
train partition = all image ids, but they should follow the label order.
"""

# Reference: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

class CustomImageDataset(Dataset):
    def __init__(
        self,
        images:list,
        img_labels:list, 
        transform:transforms=None,
        target_transform:transforms = None,
        sampler:SubsetRandomSampler=None,  
        ) -> None:

        self.img_labels = img_labels
        self.images = images
        self.transform = transform
        self.target_transform = target_transform
        self.sampler=sampler 

    def __len__(self)->int:
        """Returns the total number of images that are in the image directory."""
        return len(self.img_labels)

    def __getitem__(self, idx):
        """Get image and label based on index

        Args:
            idx (str): The image id.

        Returns:
            _type_: _description_
        """
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
        
class ImageCollection():
    def __init__(
        self, 
        img_dir:str, 
        folders:str=['train','test']
        ):

        """Class constructor"""
        self.images = None
        self.img_labels = None
        self.img_dir = img_dir
        self.folders = folders
        # key is the folder name from self.folders, starting indices of images in that folder
        self.folder_indices = {}  
        self._generate_img_indices(img_dir)

    def _generate_dataset(
        self,
        indices:list,
        transform:transforms,
        sampler:SubsetRandomSampler,
        )->CustomImageDataset:
        
        images = [ self.images[x] for x in indices]
        labels = [ self.img_labels[x] for x in indices]
        dataset = CustomImageDataset(
            images,
            labels,
            transform,
            sampler,
            )
        return(dataset)

    def generate_train_valid_dataset(
        self, 
        valid_size = 0.2,
        transform:transforms = None 
        ) -> dict:

        train_indices = self.folder_indices['train']
        indices = list(range(train_indices[0],train_indices[1]))
        np.random.shuffle(indices)
        num_train = len(indices)
        split = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_dataset = self._generate_dataset(train_idx,transform, train_sampler)
        validation_dataset = self._generate_dataset(valid_idx,transform, valid_sampler)
        result = {
            "train" : train_dataset,
            "validation" : validation_dataset
        }
        return result

    def generate_test_dataset(self, transform:transforms = None)->CustomImageDataset:
        test_indices = self.folder_indices['test']
        indices = list(range(test_indices[0],test_indices[1]))
        test_dataset = self._generate_dataset(indices,transform)
        return(test_dataset)
        
    def _generate_img_indices(self,img_dir:str):
        """Create the unique identifer for the images

        Args:
            img_dir (str): file path to image directory

        Raises:
            Exception: If images and/or labels cannot be found an error is generated.

        Returns:
            list, dictionary: Id's of image, and the index of for the labels for each image id.
        """
        if(self.check_img_dir(img_dir)):
            images = []
            labels = {}
            index = 0
            for folder in self.folders:
                root_dir = os.path.join(img_dir,folder).replace(" ","\\ ") # Mac/Linux zsh specific
                start_index = index
                
                # Image ids are the path to the image.
                for root, dirs, files in os.walk(root_dir, topdown=False):
                    for dir_name in root:    
                        for file_name in files:
                            path=(os.path.join(root, file_name.replace(" ","\\ "))) #id
                            class_label = re.sub("[0-9][0-9].",'',os.path.basename(root))
                            labels[path] = class_label
                            images.append(path)
                            index = index + 1
                self.folder_indices[folder] = (start_index, index-1)

            self.images = images
            self.img_labels = labels
        else:
            raise Exception("Unable to locate images and labels")
        return images, labels

      
    def check_img_dir(self, img_dir:str)->bool:
        """Checks that folders defined in the constructor exist in image directory

        Args:
            img_dir (str): path to the image directory.

        Returns:
            bool: True if the image directory exists and has the defined folders located in the
            directory.
        """
        if os.path.isdir(img_dir) == False:
            return(False)

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

    def get_folder_img_index_range(self, folder:str)->Tuple:
        """Returens the range of indices for the particular folder sent in."""
        return(self.folder_indices[folder])

def main():
    """Used to test the class."""
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--img_dir",required=True, help="directory path", type=str)
    #args = parser.parse_known_args()
    #cd = CustomImageDataset(args.img_dir)

    ic = ImageCollection("./project2-landmark/nd101-c2-landmarks-starter/landmark_project/landmark_images")
    data = ic.generate_train_valid_dataset(transform=transforms.CenterCrop(size=300))
    #test_data = ic.generate_test_dataset(transform=transforms.CenterCrop(300))

    num_workers = 0
    # how many samples per batch to load
    batch_size = 20 #64
    # percentage of training set to use as validation
    valid_size = 0.2

    train_loader = DataLoader(
        data['train'],
        batch_size = batch_size,
        sampler=(data['train']).sampler,
        num_workers = num_workers
    )
    
    # obtain one batch of training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    images = images.numpy()


if __name__ == "__main__":
    main()