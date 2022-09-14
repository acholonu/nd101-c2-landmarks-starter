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
import numpy as np

# Pytorch pacakges
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
# from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from pandas import DataFrame
from torch import stack

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
        self.target_labels = list(set(img_labels))
        self.images = images
        self.transform = transform
        self.target_transform = target_transform
        #self.sampler=sampler 

    def num_outputs(self)->int:
        return len(self.target_labels)

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

        image = read_image(img_path)
        print(f"Image Type: {type(image)}") # Tensor
        label = self.img_labels[idx]

        if self.transform != None:
            image = self.transform(image) #Failure
        #if self.target_transform != None:
        #    label = self.target_transform(label)
        
        #sample = {"image":image, "label":label}
        #return sample
        return image, label
        
class ImageCollection():
    def __init__(
        self, 
        img_dir:str, 
        folders:list=['train','test'],
        save_img_size:bool = True,
        min_img_size_limit:Tuple = None,
        ):

        """Class constructor.
        
            Args:
                img_dir (str): the path to folder that holds all images
                folders (str): Sub folders required in img_dir
                save_img_size (bool, optional): If True, then store image sizes in data frame.  Defaults to True
                min_img_size_limit (Tuple, optional): A Tuple that has the minium width and height for images. 
                Images must pass both the width and height limits to be included in the dataset. Set value to None if you all
                images to be include (no minimum limits on image size).  Defaults to None.
        """
        self.images:list = None
        self.img_labels:list = None
        self.save_img_size:bool = save_img_size
        self.img_sizes:list = []
        self.img_dir:str = img_dir
        self.folders:list = folders
        self.min_img_size_limit = min_img_size_limit
        # key is the folder name from self.folders, starting indices of images in that folder
        self.folder_indices = {}  
        self._generate_img_indices(img_dir)

    def verify_image(
            self, 
            filename:str, 
            check_image_variance:bool = False,
        )->bool:
        """_summary_

        Args:
            filename (str): the image file to open
            check_image_variance (bool, optional): Set to True if you want to check if images
            has little to no variance in pixel values. So if the picture is all black verify_image would
            return False for that image. Defaults to False.

        Returns:
            bool: Returns true if the image checks all pass.
        """
        try:
            img = Image.open(filename) # open the image file
            img.verify() # verify that it is, in fact an image without loading the image.  It errors if there is a probelem.
            # check image has variance in data.
            if self.save_img_size == True:
                width, height = img.size
                if self.min_img_size_limit is not None:
                    if width < self.min_img_size_limit[0]:
                        print(f"Image: {filename} width:{width} is less than limit:{self.min_img_size_limit[0]}.")
                        return(False)
                    elif height < self.min_img_size_limit[1]:
                        print(f"Image: {filename} height:{height} is less than limit:{self.min_img_size_limit[1]}.")
                        return(False)
                row = [filename, width, height]
                self.img_sizes.append(row)

            if check_image_variance:
                result = self._validate_img_variance(img)
                if result == False:
                    print(f"No Pixel Variance for file: {filename}")
                    return(False)
            img.close() #reload is necessary in my case
            return(True)
        except (IOError, SyntaxError) as e:
            print(f'Bad file: {filename}. Error: {e}') # print out the names of corrupt files
            return(False)

    def _validate_img_variance(self, img:Image)->bool:
        # measure of information complexity in the image. Range: 0 to 1, where zero means
        # very little complexity (very uniform)
        entropy = img.entropy()  
        print(entropy)
        # TODO: Not sure if I need this function.  Cool experimentation
        return True

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

    def get_img_sizes(self)->DataFrame:
        if self.img_sizes is None: return(self.img_sizes)
        df = DataFrame(self.img_sizes, columns=['filename','width','height'])
        return(df)

    def describe_img_sizes(self)->DataFrame:
        df = self.get_img_sizes()
        if df is None: return(df)
        return(df.describe())

def test(use_five_crop:bool = False):
    """Used to test the class."""
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--img_dir",required=True, help="directory path", type=str)
    #args = parser.parse_known_args()
    #cd = CustomImageDataset(args.img_dir)
    transform = None
    num_workers = 0
    # how many samples per batch to load
    batch_size = 20 #64
    # percentage of training set to use as validation
    valid_size = 0.2

    ic = ImageCollection(
        "./project2-landmark/nd101-c2-landmarks-starter/landmark_project/landmark_images",
        min_img_size_limit=(256,256),
        )

    # REFERENCE: https://pytorch.org/vision/main/transforms.html#compositions-of-transforms
    # REFERENCE: https://pytorch.org/vision/stable/generated/torchvision.transforms.FiveCrop.html
    # Because FiveCrop creates a tuple with the 5 crops, you need to figure
    # out a way to convert to a 4D tensor (width,height,color times 5), and how to parse each 
    # image.
    if use_five_crop:
        mean=(0.5,0.5,0.5)
        std=(0.5,0.5,0.5)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.ColorJitter(brightness=.5,hue=.3,contrast=.5,saturation=.2),
            transforms.RandomRotation(degrees=(0,180)), #randomly rotate between 0 and 180 degrees
            transforms.FiveCrop(size=(100,100)), # crops the given image into four corners and the central crop, returns a tuple
            transforms.Lambda(lambda crops: stack([transforms.PILToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: stack([crop.type(torch.float) for crop in crops])),
            transforms.Lambda(lambda crops: stack([transforms.Normalize(mean, std)(crop) for crop in crops])) #NOT WORKING

            #Below unpacks the FiveCrop line (266) above.  Just added normalization
            #transforms.Lambda(lambda crops: [transforms.PILToTensor()(crop) for crop in crops]), #  applies a user-defined lambda as a transform.  Needed for FiveCrop
            #transforms.Lambda(lambda crops: [crop.type(torch.float) for crop in crops]),
            #transforms.Lambda(lambda crops:[transforms.Normalize(mean, std)(crop) for crop in crops]), #THis is not working
            #transforms.Lambda(lambda crops: stack(crops)) #returns a 4D tensor [batch, feature_maps, width, height],
        ])
    else:
        # Simplified transform - This works
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ColorJitter(brightness=.5,hue=.3,contrast=.5,saturation=.2),
            transforms.RandomRotation(degrees=30), #randomly rotate between 0 and 180 degrees
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    data = ic.generate_train_valid_dataset(
            valid_size = valid_size,
            transform=transform
        )
    
    train_loader = DataLoader(
        data['train'],
        batch_size = batch_size,
        num_workers = num_workers
    )

    # obtain one batch of training images
    if use_five_crop:
        # Have to do this method because I am using FiveCrop
        # Reference: https://pytorch.org/vision/stable/generated/torchvision.transforms.FiveCrop.html
        # Reference: https://stackoverflow.com/questions/62827282/how-to-handle-transforms-fivecrop-change-in-tensor-size
        for batch_idx, (data,target) in enumerate(train_loader):
            # ncrops = number of crops, which our situation is 5
            batch_size, ncrops, colors, height, width = data.size() 
            # Let's view the break down
            print(f'Data: {data.view(-1, colors, height, width)}, Data Type: {data.dtype}')

            # in the training process, you would put each cropped image in the model for training
            # Then you would aggregate the the results for each of the 5 (i.e., ncrops) crops that made up
            # the original image.

            # Looks like you can handle augmentation in two ways.
            #   
            # 1) Augment the dataset before doing the data loader
            # transforming the dataset first and then put it the resulting augmentation into a dataset object.  
            # And then the dataset object is passed into the data loader.
            # SEE example: EXAMPLE 11 (https://www.programcreek.com/python/example/117701/torchvision.transforms.TenCrop)
            # go to the bottom of the page.
            #
            # 2) Applying the transformation and then fusing everything back together after applying transformation
            
            # NEXT STEPS plot the five crop before and after fusing together image for 1 image.  So I can understand
            # what is happening

            # Early stopping of loop
            if batch_idx > 1:
                break
    else:
        # No Data Augmentation (Adding more images vs. modifying images), so can use the simpler method
        dataiter = iter(train_loader)
        images, labels = dataiter.next()
        images = images.numpy()
        print(labels)

    # Describe Image Sizes
    df = ic.describe_img_sizes()
    print(df)
    df.to_csv("describe_img_sizes.csv")
    print("done")

if __name__ == "__main__":
    test()