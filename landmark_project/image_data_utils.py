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
from PIL import Image
from typing import Tuple

# Data wrangling and visualization
from matplotlib import transforms
import numpy as np

# Pytorch pacakges
from torchvision.io import read_image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
# from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

"""So I need to flatten the dataset.  for labels->dictionary key=image id, value = label, 
train partition = all image ids, but they should follow the label order.
"""

# Reference: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

class CustomImageDataset(Dataset):
    """Dataset stores the samples and their corresponding labels."""
    def __init__(
            self,
            images:list,
            img_labels:list, 
            transform:transforms=None,
            target_transform:transforms = None,
            #sampler:SubsetRandomSampler=None,  # Why doesn't this work?
        ) -> None:

        self.img_labels = img_labels
        self.images = images
        self.transform = transform
        self.target_transform = target_transform
        #self.sampler=sampler 

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
        img_path = self.images[idx]
        print(f"image path: {img_path}")

        image = read_image(img_path)
        label = self.img_labels[idx]
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
        folders:list=['train','test']
        ):

        """Class constructor"""
        self.images:list = None
        self.img_labels:list = None
        self.img_dir:str = img_dir
        self.folders:list = folders
        # key is the folder name from self.folders, starting indices of images in that folder
        self.folder_indices = {}  
        self._generate_img_indices(img_dir)

    def verify_image(self,filename:str)->bool:
        try:
            img = Image.open(filename) # open the image file
            img.verify() # verify that it is, in fact an image
            img.close() #reload is necessary in my case
            return(True)
        except (IOError, SyntaxError) as e:
            print('Bad file:', filename) # print out the names of corrupt files
            return(False)

    def _generate_dataset(
        self,
        indices:list,
        transform:transforms,
        target_transform:transforms = None
        )->CustomImageDataset:
        
        images = [ self.images[x] for x in indices]
        labels = [ self.img_labels[x] for x in indices]
        dataset = CustomImageDataset(
                images,
                labels,
                transform,
                target_transform,
                #sampler
            )
        return(dataset)

    def generate_train_valid_dataset(
        self, 
        valid_size = 0.2,
        transform:transforms = None 
        ) -> dict:

        train_indices = self.folder_indices['train']
        indices = list(range(train_indices[0],train_indices[1]+1))
        np.random.shuffle(indices)
        num_train = len(indices)
        split = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        
        # Samples elements randomly from a given list of indices, without replacement.
        # Causes failure and not needed since I already randomize
        #train_sampler = SubsetRandomSampler(train_idx)
        #valid_sampler = SubsetRandomSampler(valid_idx)

        train_dataset = self._generate_dataset(train_idx,transform)
        validation_dataset = self._generate_dataset(valid_idx,transform)
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
            labels = []
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
                            if self.verify_image(path) == True:
                                labels.append(class_label)
                                images.append(path)
                                index = index + 1
                self.folder_indices[folder] = (start_index, index-1)
                print(f"Indicies for {folder} are: {self.folder_indices[folder]}")
            
            self.images = images.copy()
            self.img_labels = labels.copy()
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

def test():
    """Used to test the class."""
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--img_dir",required=True, help="directory path", type=str)
    #args = parser.parse_known_args()
    #cd = CustomImageDataset(args.img_dir)

    num_workers = 0
    # how many samples per batch to load
    batch_size = 20 #64
    # percentage of training set to use as validation
    valid_size = 0.2

    ic = ImageCollection("./project2-landmark/nd101-c2-landmarks-starter/landmark_project/landmark_images")

    # Desired Transform - Does not work.  Need to read docs
    transform = transforms.Compose([
        transforms.ToPILImage(), # This is needed for the ColorJitter
        transforms.FiveCrop(size=(100,100)), # crops the given image into four corners and the central crop
        transforms.ColorJitter(brightness=.5,hue=.3,contrast=.5,saturation=.2),
        transforms.RandomRotation(degrees=(0,180)), #randomly rotate between 0 and 180 degrees
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Simplified transform - This works
    transform_img = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    data = ic.generate_train_valid_dataset(
            valid_size = valid_size, 
            transform=transform_img
        )
    #test_data = ic.generate_test_dataset(transform=transforms.CenterCrop(300))

    train_loader = DataLoader(
        data['train'],
        batch_size = batch_size,
        num_workers = num_workers
    )

    # obtain one batch of training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    images = images.numpy()
    print(labels)

if __name__ == "__main__":
    test()