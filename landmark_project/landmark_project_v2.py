import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms #toTensor is the function you want to use
#from torchvision.transforms import Lambda
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from PIL import Image

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
                    image_input_size = 32, #Size of final images from feature functions
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
                    ('fc1', nn.Linear(image_input_size * image_input_size, num_hidden1_nodes)), # fully connected hidden layer 1
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
        print(f"Foward: x.size() = {x.size()}")  # [64, 3, 256, 256]
        inputs = self.features(x)
        print(f"inputs.size() = {inputs.size()}") # [6, 64, 1024]
        result = self.model(inputs) # Original
        return result


def verify_image(
            filename:str, 
            min_img_size_limit=(256,256),
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
            
            width, height = img.size
            if min_img_size_limit is not None:
                if width < min_img_size_limit[0]:
                    return(False)
                elif height < min_img_size_limit[1]:
                    return(False)            
            img.close() #reload is necessary in my case
            return(True)
        except (IOError, SyntaxError) as e:
            print(f'Bad file: {filename}. Error: {e}') # print out the names of corrupt files
            return(False)


def create_dataset(
        img_dir, 
        n_outputs, 
        batch_size:int,
        num_workers:int,
        shuffle:bool = True, 
        valid_size:float = 0.20
    )->dict:
    transform_img, target_transform = get_transforms(n_outputs)
    img_dataset = datasets.ImageFolder(os.path.join(img_dir,'train'),transform=transform_img, target_transform=target_transform, is_valid_file=verify_image)
    test_dataset = datasets.ImageFolder(os.path.join(img_dir,'test'),transform_img, target_transform, is_valid_file=verify_image)

    indices = list(range(0,len(img_dataset)))
    np.random.shuffle(indices)
    num_train = len(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_dataset = Subset(img_dataset, train_idx)
    valid_dataset = Subset(img_dataset, valid_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    result = {
        "train":train_loader,
        "valid":valid_loader,
        "test":test_loader
    }
    return result


def get_transforms(n_outputs:int)->transforms.Compose:

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
    
    #one hot encoded tensor for label.  Assuming label is the numerical index of the label.
    #Reference: https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html#lambda-transforms
    #target_transform = Lambda(lambda y: torch.zeros(n_outputs, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    target_transform = None # Testing
    return(transform_img, target_transform)


def check_gpu()->bool:
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
        #device = "cpu"
    else:
        print('CUDA is available!  Training on GPU ...')
        #device = "cuda"
    #print(f"Using {device} device")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return(device)


def get_optimizer_scratch(model, learning_rate = 0.001):
    # Reference about Adam optimizer: https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
    # Pytorch Reference: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
    #learning_rate = 0.001 #This is the default learning_rate for the Adam algoritm anyway
    optimizer = torch.optim.Adam(model.parameters(), learning_rate) # The Computational approach to doing optimization with the set criterion
    return(optimizer)


def train(
        n_epochs, 
        loaders, 
        model, 
        optimizer, 
        criterion, 
        save_path:str = None
    )->Net:
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
        model.train() # Inherited from nn.Module class. set the module to training mode
        for batch_idx, (X, y) in enumerate(loaders['train']):
            print(f"spot 2 - batch_idx: {batch_idx}")        

            optimizer.zero_grad() # clear the gradients of all optimized variables
            output = model(X)  # forward pass: compute predicted outputs of the batch by passing inputs to the model
            loss = criterion(output, y) # calculate the batch loss
            
            # loss.item() extracts the batch loss value as a float, but cross_entropy divides this loss by the num_elements
            # This is why the train_loss equation is defined in this manner.
            # Reference: https://stackoverflow.com/questions/61092523/what-is-running-loss-in-pytorch-and-how-is-it-calculated
            train_loss += loss.item()*X.size(0)
            loss.backward() # Locating the weights that produce the most error.  Backward propagating the error
            optimizer.step() # perform a single optimization step (parameter update), updating weights

        if train_loss == np.nan:
            raise ValueError("train loss is nan")

        ######################    
        # validate the model #
        ######################
        # set the model to evaluation mode
        model.eval()
        # For Loop: Sums all the loss across the batch items
        for batch_idx, (X, y) in enumerate(loaders['valid']):
            output = model(X) # forward pass: compute predicted outputs by passing inputs to the model
            loss = criterion(output, y) # calculate the batch loss 
            valid_loss += loss.item()*X.size(0) # update average validation loss, running loss

        if valid_loss == np.nan:
            raise ValueError("valid loss is nan")

        ## TODO: find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data.item() - train_loss))
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
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
        
    return model


def main():
    num_workers = 0 # Final
    batch_size = 64 #  the number of data samples propagated through the network before the parameters are updated
    n_outputs = 50
    valid_size = 0.2 # percentage of training set to use as validation
    n_epochs = 100 #the number times to iterate over the dataset

    img_dir = "./project2-landmark/nd101-c2-landmarks-starter/landmark_project/landmark_images"

    print("Starting ....")
    print("Creating dataset...")

    loaders_scratch = create_dataset(img_dir=img_dir, n_outputs=n_outputs, batch_size=batch_size, num_workers=num_workers, valid_size=valid_size)
    device = check_gpu()
    criterion_scratch = nn.CrossEntropyLoss()

    # instantiate the CNN
    print("Creating model architecture...")
    model_scratch = Net(n_outputs=n_outputs).to(device) # move tensors to GPU if CUDA is available
    
    print("Training...")
    optimizer_scratch = get_optimizer_scratch(model_scratch)
    model_transfer = train(n_epochs, loaders_scratch, model_scratch, optimizer_scratch, criterion_scratch, "model_transfer.pt")
    print("Training Complete...")

if __name__ == "__main__":
    main()