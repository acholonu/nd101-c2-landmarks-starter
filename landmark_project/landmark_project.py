import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms #toTensor is the function you want to use
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch import Tensor
from torch import stack

from image_data_utils import ImageCollection

class PrintSize(nn.Module):
  def __init__(self):
    super(PrintSize, self).__init__()
    
  def forward(self, x):
    print(x.shape)
    return x

    
# define the CNN architecture
class Net(nn.Module):
    ## TODO: choose an architecture, and complete the class
    def __init__(
                    self,
                    n_outputs = 50,
                    image_input_size = 64, #Size of final images from feature functions
                    num_hidden1_nodes = 64,
                    num_hidden2_nodes = 50, 
                ):
        """Initializing the neural network

        In Pytorch, to define any neural network you have to name and define any layers that have any
        learned weight values. 

        Args:
            n_outputs (int, optional): The number of target labels. Defaults to 50.
        """
        super(Net, self).__init__()

        num_output_nodes = n_outputs
        
        ## Define layers of a CNN
        ## Reference: https://towardsdatascience.com/pytorch-how-and-when-to-use-module-sequential-modulelist-and-moduledict-7a54597b5f17
        self.features = nn.Sequential(
            OrderedDict(
                [   
                    ('printsize_0', PrintSize()),
                    ('conv_layer1',nn.Conv2d(3, 16, 3, padding=1)), #depth is 3, size = (256,256)
                    # Relu activation function?
                    ('relu1', nn.ReLU(True)), # activation function. Makes sure values are in a consistent range, True means do in place
                    ('printsize_1', PrintSize()),
                    ('conv_layer2',nn.Conv2d(16, 32, 3, padding=1)), #depth is 16, size = (256,256), padding=1, means put a border of zeroes, 1 pixel wide around the image. Instead of zeros, I could use the strategy of nearest value.
                    ('printsize_2', PrintSize()),
                    ('maxpooling_layer1',nn.MaxPool2d(2,2)), #size = (128,128)
                    ('relu2', nn.ReLU(True)), # activation function. Makes sure values are in a consistent range
                    #depth = 32, also cut the resultant filtered image in half with stride =2
                    ('printsize_3', PrintSize()),
                    ('conv_layer3',nn.Conv2d(32, 64, 3, padding=1, stride=2)), #depth=64, size=(128,128)
                    ('printsize_4', PrintSize()),
                    ('maxpooling_layer2',nn.MaxPool2d(2,2)), #depth=64, size=(64,64)
                    ('relu3', nn.ReLU(True)), # activation function. Makes sure values are in a consistent range
                    ('printsize_5', PrintSize()),
                    ('flatten_layer', nn.Flatten()),
                    ('printsize_6', PrintSize()),
                ]
            )
        )

        # Normal Neural Network
        
        ## Define Layers for the Linear Neural Network
        ## Need to determine the final size from the convolution
        self.model = nn.Sequential(
            OrderedDict(
                [
                    ('fc1', nn.Linear(image_input_size, num_hidden1_nodes)), # fully connected hidden layer 1
                    ('dropout1', nn.Dropout(.2)), # dropout layer to help reduce overfitting
                    ('relu1', nn.ReLU()), # activation function. Makes sure values are in a consistent range
                    ('fc2', nn.Linear(num_hidden1_nodes, num_hidden2_nodes)), # fully connected hidden layer 
                    ('dropout2', nn.Dropout(.2)), # to dropout layer to help reduce overfitting
                    ('relu2',nn.ReLU()), # activation function.  Makes sure values are in a consistent range
                    ('fc3', nn.Linear(num_hidden2_nodes, num_output_nodes)), # output layer
                    #('softmax', nn.Softmax(dim=1)) # class probabilities.  Don't need this because I am using cross-entropies optimization
                ]
            )
        )          

    def forward(self, x):
        ## Define forward behavior 
        print(f"Foward: x.size() = {x.size()}")  
        inputs = self.features(x)
        print(f"inputs.size() = {inputs.size()}")
        inputs_view = inputs.view(-1, inputs.size(0))
        print(f"inputs_view.size(): {inputs_view.size()}")
        #x = x.view(-1, self.image_input_size * self.image_input_size)
        result = self.model(inputs_view)
        return result


def create_datasets(img_dir, min_img_size_limit=(256,256))->dict:
    ic = ImageCollection(img_dir, min_img_size_limit=min_img_size_limit)
    print(f"Number Filtered: {ic.get_number_filtered()} Remaining images:{ic.get_number_images()}")
    
    df = ic.describe_img_sizes()
    print(df)
    
    transform_img = get_transform()
    dataset = ic.generate_train_valid_dataset(transform=transform_img)
    test_data = ic.generate_test_dataset(transform=transform_img)
    num_images = len(ic.img_labels)
    num_labels = (dataset['train']).num_outputs()
    print(f"Number of Labels: {num_labels}\nNumber of Images: {num_images}")
    
    data ={
        "train":dataset["train"],
        "valid":dataset["validation"],
        "test":test_data
    }
    return data


def create_dataloaders(
        data:dict,
        batch_size:int,
        num_workers:int,
        shuffle:bool = True,
    )->dict:
    """DataLoader wraps an iterable around the Dataset to enable easy access to the samples."""

    train_loader = torch.utils.data.DataLoader(
        data['train'],
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = shuffle,
    )

    valid_loader = torch.utils.data.DataLoader(
        data['valid'],
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = shuffle
    )

    test_loader = torch.utils.data.DataLoader(
        data['train'], 
        batch_size=batch_size, 
        num_workers=num_workers
    )
    loaders_scratch = {
        'train': train_loader, 
        'valid': valid_loader, 
        'test': test_loader
    }
    return(loaders_scratch)


def get_transform()->transforms.Compose:
    mean=[0.5,0.5,0.5]
    std=[0.5,0.5,0.5]
    image_size = 256


    # REFERNCE: https://pytorch.org/vision/main/transforms.html#compositions-of-transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256), #resize all images so that the smallest side = 256 pixels
        transforms.ColorJitter(brightness=.5,hue=.3,contrast=.5,saturation=.2),
        transforms.RandomRotation(degrees=(0,180)), #randomly rotate between 0 and 180 degrees
        transforms.FiveCrop(size=(100,100)), # crops the given image into four corners and the central crop, returns a tuple
        transforms.Lambda(lambda crops: stack([transforms.PILToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: stack([crop.type(torch.float) for crop in crops])),
        transforms.Lambda(lambda crops: stack([transforms.Normalize(mean, std)(crop) for crop in crops])) #NOT WORKING
    ])

    # Simplified transform - This works
    transform_img = transforms.Compose([
        transforms.ToPILImage(), # Convert image data to PIL data type
        transforms.Resize((256,256)), #resize all image down to 256 X 256 pixels
        #transforms.CenterCrop(image_size), # center crop resultant image to be 224 X 224 image
        transforms.ColorJitter(brightness=.1,hue=.1,contrast=.1,saturation=.1),
        transforms.RandomRotation(degrees=15), #randomly rotate between 15 and -15 degrees
        transforms.ToTensor(), # Convert image data to Tensor data type
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize the data to have a mean of .5 and standard deviation .5)
        ])
    return(transform_img)


def check_gpu()->bool:
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
        use_cuda = False
    else:
        print('CUDA is available!  Training on GPU ...')
        use_cuda = True
    return(use_cuda)

def get_optimizer_scratch(model, learning_rate = 0.001):
    ## TODO: select and return an optimizer
    
    # specify optimizer (stochastic gradient descent) and learning rate = 0.01
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Reference about Adam optimizer: https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
    # Pytorch Reference: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
    #learning_rate = 0.001 #This is the default learning_rate for the Adam algoritm anyway
    optimizer = torch.optim.Adam(model.parameters(), learning_rate) # The Computational approach to doing optimization with the set criterion
    return(optimizer)

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        # set the module to training mode
        model.train() # Inherited from nn.Module class
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                print(f"spot 2 - batch_idx: {batch_idx}")
                data, target = data.cuda(), target.cuda()

            ## TODO: find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data.item() - train_loss))
            
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model.forward(data)
            # calculate the batch loss
            loss = criterion(output, target) #TODO: ERROR HERE.  Target must be a tensor
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            
            # update training loss [Ugochi: Note sure if I did this right]
            train_loss += loss.item()*data.size(0)

        if train_loss == np.nan:
            raise ValueError("train loss is nan")

        ######################    
        # validate the model #
        ######################
        # set the model to evaluation mode
        model.eval()
        # For Loop: Sums all the loss across the batch items
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                print(f"spot 3 - batch_idx: {batch_idx}")
                data, target = data.cuda(), target.cuda()
                print(f"batch_idx: {batch_idx} target: {target}")

            ## TODO: update average validation loss 
             # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss 
            valid_loss += loss.item()*data.size(0)

        if valid_loss == np.nan:
            raise ValueError("valid loss is nan")
        train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data.item() - train_loss))
        valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data.item() - valid_loss))
            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))

        ## TODO: if the validation loss has decreased, save the model at the filepath stored in save_path
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss)
            )
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
        
    return model


def main():
    #num_workers = 0 # Testing
    #batch_size = 6 # Testing
    #n_outputs = 1 # Testing

    num_workers = 6 # Final
    batch_size = 64 # # how many samples per batch to load
    n_outputs = 50

    # percentage of training set to use as validation
    valid_size = 0.2
    n_epochs = 20

    #img_dir = "/Users/ujones/Dropbox/Data Science/Python Project Learning/Udacity/Deep Learning/project2-landmark/nd101-c2-landmarks-starter/landmark_project/images" # For Testing
    #img_dir = "/Users/ujones/Dropbox/Data Science/Python Project Learning/Udacity/Deep Learning/project2-landmark/nd101-c2-landmarks-starter/landmark_project/landmark_images" #Real Batch
    img_dir = "./project2-landmark/nd101-c2-landmarks-starter/landmark_project/images"
    #img_dir = "./project2-landmark/nd101-c2-landmarks-starter/landmark_project/landmark_images"

    print("Starting ....")
    print("Creating dataset...")
    data = create_datasets(img_dir=img_dir)
    loaders_scratch = create_dataloaders(
                        data,
                        batch_size,
                        num_workers,
                        True,
                    )
    use_cuda = check_gpu()
    criterion_scratch = nn.CrossEntropyLoss()
    # instantiate the CNN
    print("Creating model architecture...")
    model_scratch = Net(n_outputs=n_outputs) # Testing

    # move tensors to GPU if CUDA is available
    if use_cuda:
        print(f"spot 1 - batch_idx: {model_scratch}")
        model_scratch.cuda()
    
    optimizer_scratch = get_optimizer_scratch(model_scratch)
    print("Training...")
    model_transfer = train(n_epochs, loaders_scratch, model_scratch, optimizer_scratch, criterion_scratch, use_cuda, "model_transfer.pt")
    print("Training Complete...")


if __name__ == "__main__":
    main()