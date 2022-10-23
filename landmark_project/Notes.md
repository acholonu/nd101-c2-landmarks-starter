## What Am I really Learning

If I think about the larger picture of machine learning learning, this project is really about learning from images. All machine learning learns from examples, so the format of examples that you are using requires different processing to find the final features.  Often when we first start learning machine learning, the focus is really on tabular data (think excel files or .csv files).  Focusing on tabular data at first is important because it remove the need to focus on data processing (outside of cleaning) and feature engineering because many approaches, at least for data setup as cross-sectional, you can easily derive the features (e.g., sums of occurence, average, value at end of the year).  But even tabular data can get complicated, for example, when we think about data as nested or as a time series.  Other formats that we learn from are audio, video, text, and sensors.  

So as a data scientist you are specializing in formats of data that you can create models on, the domain you are applying the machine learning processes to, and the algorithms you are using to learn from examples.

## For Images

Above data frame show the results of filtering images who fail a minimum width and height size of (256X256).  Most images have a size of 800 x 600.  Max size is 800 x 800.  I wanted to reduce the amount of distortion that will happen when I apply the resize transform.

## Loading Data

1. Transform data.  Great Reference for possible transformation: https://pytorch.org/vision/main/auto_examples/plot_transforms.html?highlight=transforms
2. Create the data loader
3. Create the train, validation, and test datasets

I am finding that understanding the image format that each transform expects is key to avoiding errors. So some transforms (e.g., Resize) expect PIL format others (e.g., Normalize) expect Tensor format.

### [Side Note] Data Normalization

Data normalization is an important pre-processing step. It ensures that each input (each pixel value, in this case) comes from a standard distribution. That is, the range of pixel values in one input image are the same as the range in another image. This standardization makes our model train and reach a minimum error, faster!

Data normalization is typically done by subtracting the mean (the average of all pixel values) from each pixel, and then dividing the result by the standard deviation of all the pixel values. Sometimes you'll see an approximation here, where we use a mean and standard deviation of 0.5 to center the pixel values.

### Filtering non-informative images?

Looking at the images (e.g., Taj_Mahal), I found some pictures may have very little or no information.  So for example, all pixels are the same.  I should probably filter this out because it will reduce noise.  This should be true for all images (train and test).  So is there a way for me to create a cleaner that remove images with little variation?  Or will the neural network handle this?  

The network would possible figure this out.  But I could reduce processing time by removing?  I guess this is not a priority.  
---

### Vocab and Design Decisions

The problem: The word `Loss` is used in many ways.  So below I am identifying the conceptual differences of the use of the term `loss`.

#### The various Scoring and Loss Functions

- **scoring function** - the score that the base model f(x,W) = Wx+ b gives a point.  Where W are the weights, and x is the the example from the dataset X.
- **error/Loss function** - is used for evaluating the error of an **indivdual train example (row)**. It is the penalty you attribute to error, where error is defined as ypred-yactual.
- **Cost/Loss function** - is used for calculating the average loss over the entire training set.  I like using the term cost function.
- **Scoring(h) function for hypertuning** - The metric that your fitting method optimizes for a given model with all hyperparameters set. The metric used to choose between your optimized model (i.e. how you pick the best hyperparameters). (So for GridSearchCV).

If you are trying to minimize the MAE, you would ideally want to have MAE as your cost function (so each model has the smallest possible MAE, given the hyperparameters) and have MAE as your Scoring(h) function (so you pick the best hyperparameters). If you use MSE as your cost function, and MAE as your scoring(h), you are unlikely to find the best answer.

[Cost/Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)

### Optimizers

When training a model we try to minimize the Cost function to select the best **weights** for the training dataset.  Note, while finding the best **weights** for a training set, the hyperparameters linked to the selected algorithm (e.g., Random Forest, Multinormal regression, SVM), stays constant. Hyperparameter tuning is where we find multiple models using the the same training set (so Cost function to optimize weights still happen), but change the hyperparameter (e.g., Grid Search). 

Again, for optimization, we want to minimize the chosen Cost/Loss function. In calculus, to locate the mim/max you would find the point where the derivative is zero.  However, in most cases, it is really hard to compute the derivative of a Loss function because of the high dimensionality of the function (all those columns from the dataset!). Therefore, we use computational numerical analysis approaches to iteratively find the minimum (so approximate the derivative). There are many optimizers available that simulate the derivative in various manners. Additionally, optimizers is a computational process (e.g., gradient descent, stochastic gradient descent); thus, they have differing ways they use memory (what they remember as they do the simulation), cpu usage (how they do computation), speed (parallel or serial processes), and algorithmic behaviors to find the minimum. And each are better for different model algorithms (e.g., Regression, XGBoost, Random Forest). So there are design decisions here.

Types of Optimizers:

- Gradient Descent
- Stochastic Gradient Descent
- [Overview of Optimizers](./references/Overview_of_Optimizers.pdf)
- [Pytorch Optimzers](https://pytorch.org/docs/stable/optim.html#algorithms)

### Regularization

When we optimize, we are only using 1 dataset to find weights. However, the overall goal, is to build a model that can generalize to data that it has never seen. Sometimes, optimizer may be too good in finding the best model for the particular training dataset.  So the model performs well on training set, but when applied to the validation or test set, the performance significantly differs (i.e., Overfitting). Often when you overfit, the optimizer has created a model that is overly complex, in order to fit the specific data points of the training set.

Regularization tries to prevent the optimization process from overfitting. Regularization introduces randomness and penalties around complex into the optimization process. So now the optimizer is not just looking for the best weights for the training dataset but has to also balance that with the reduction of complexity. So we have basically added another constraint for the optimizer to take into consideration when optimizing.  This term is really about model generalization. There are many types of Regularization.

Types of Regularization -

- L1
- L2 -
- Dropout - neural networks.  Randomly drop out nodes during training
- Early Stopping - For example, with Random Forest, you give it a minimal node size (so a node cannot have less than 20 values).  Or tree can only branch for this many levels.
- Data Augmentation - Image manipulations (rotations, color changes, five crop).  Adding variation to the training dataset, so not to optimize to the training set. 
- Batch Normalization
- [Regularization Article](https://medium.com/analytics-vidhya/understanding-regularization-with-pytorch-26a838d94058)

### Performance Metrics

- [Classifier Performance metrics](https://scikit-learn.org/stable/modules/classes.html?highlight=sklearn%20metrics#classification-metrics)
- [Regression Performance metrics](https://scikit-learn.org/stable/modules/classes.html?highlight=sklearn%20metrics#regression-metrics)
- [Multi-Label Ranking metrics](https://scikit-learn.org/stable/modules/classes.html?highlight=sklearn%20metrics#multilabel-ranking-metrics)
- [Clustering Metrics](https://scikit-learn.org/stable/modules/classes.html?highlight=sklearn%20metrics#clustering-metrics)

---

### Define the Network [Architecture](http://pytorch.org/docs/stable/nn.html)

This time, you'll define a CNN architecture. Instead of an MLP, which used linear, fully-connected layers, you'll use the following:
* [Convolutional layers](https://pytorch.org/docs/stable/nn.html#conv2d), which can be thought of as stack of filtered images. Convolutional layer consists of filters and feature maps. Where filters carry the input weights and the **feature map** is the output according to the weight applied on the filter.
* [Max Pooling layers](https://pytorch.org/docs/stable/nn.html#maxpool2d), which reduce the x-y size of an input, keeping only the most _active_ pixels from the previous layer. Max-pooling is a form of Regularization.  It generalizes the data by selecting the max value of a filter region and using that as the representative. There are other Pooling Layers, such as averaging the pixel (i.e., Average Pooling Layers), but this results in smoothing versus high contrast. The basic work of the pooling layer is to downsample the **feature map**. Good Reference: <https://analyticsindiamag.com/comprehensive-guide-to-different-pooling-layers-in-deep-learning/>

* ![Pooling Layer](./references/pooling_layers.png)

* ![Max Pooling Layer](./references/max_pooling_ex.png)  Shows an example of what max pooling does.

* The usual Linear + Dropout layers to avoid overfitting and produce a 10-dim output.

A network with 2 convolutional layers is shown in the image below and in the code. Note: the CNN is going to pull out the relevant "features" for distinguishing for identifying the different classes (locations).

<img src='references/2_layer_conv.png' height=80% width=80% />

![Another example of CNN](./references/cnn.png)

The more convolutional layers you include, the more complex patterns in color and shape a model can detect. It's suggested that your final model include 2 or 3 convolutional layers as well as linear layers + dropout in between to avoid overfitting. 

It's good practice to look at existing research and implementations of related models as a starting point for defining your own models. You may find it useful to look at [this PyTorch classification example](https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py) or [this, more complex Keras example](https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py) to help decide on a final structure.

#### Output volume for a convolutional layer

To compute the output size of a given convolutional layer we can perform the following calculation (taken from [Stanford's cs231n course](http://cs231n.github.io/convolutional-networks/#layers)):
> We can compute the spatial size of the output volume as a function of the input volume size (W), the kernel/filter size (F), the stride with which they are applied (S), and the amount of zero padding used (P) on the border. The correct formula for calculating how many neurons define the output_W is given by `((W−F)+(2P))/S)+1`. 

For example for a 7x7 input and a 3x3 filter with stride 1 and pad 0 we would get a 5x5 output, `((7-3) + (2*0))/1) + 1 = 5`. With stride 2 we would get a 3x3 output, `((7-3) + (2*0))/2) + 1 = 3`.

#### Convolution Layer in pytorch

![Pytorch Convolution Layer Documentation](./references/pytorch_convolution_layers.png)

#### More Vocab

- **Filter:** A set of weights in a 2D format.  The weights are defined in a way to identify patterns of intensity changes, particularly around edges. High Pass Filter, Low Pass Filter.  Filters are sometime referred to as **Kernels**.
- **Stride:** How many pixels to jump as you move the filter over the image before it does the convolution. Stride =1, means move the filter 1 over before applying the convolution. So you are not skipping any pixels. Reduces the size (x,y) of the final filtered image.  So if stride is two, the dimensions of the final image will be reduced by half.  I see this as another way of generalizing your network.
- **Edge handling:** Kernel convolution relies on centering a pixel and looking at it's surrounding neighbors. So, what do you do if there are no surrounding pixels like on an image corner or edge? Well, there are a number of ways to process the edges, which are listed below. It’s most common to use padding, cropping, or extension. In extension, the border pixels of an image are copied and extended far enough to result in a filtered image of the same size as the original image.
- **Padding:** The image is padded with a border of 0's, black pixels. An alternative value for the border is the nearest values.  So there is even flexibility when it comes to how you pad.

- **Convolution:** See image below.  It a process for applying a filter. Steps: Center a filter full of weights on a pixel. Multiple the weights of a filter on the pixel values that it overlaps.  Then sum up the values to get the new value for the center pixel.

<img src='references/convolution_layers.gif' height=50% width=50% />

Another Good Reference about designing CNNs: <https://cs231n.github.io/convolutional-networks/#conv>

---

## Connect to EC2

I am connecting to EC2 in order to run on a GPU.  Please see below on how to set it up. I am going to try the lower EC2 instance type: `g4dn.xlarge`. It is an Intel based server. Cost is $0.51 per hour on a Linux On-Demand machine. The price is current as of 10/1/2022. The instance has **1 GPU**, **4vCPU**, **16GB RAM**, **45 GB** storage, Moderate Network Performance.

Reference: <https://medium.com/@christyjacob4/using-vscode-remotely-on-an-ec2-instance-7822c4032cff>

Steps To Connecting Remotely Using VSCode:

1. Make sure **Remote-SSH** from Microsoft extension is installed for VSCode on your local machine.
2. Use the AWS calculator: <https://calculator.aws/>, to estimate your usage cost.  Pretty cool.  It estimated that if I used the selected instance for 7 hours a week (so I would need to turn off the instance) for 4 weeks, it would cost me $23.15 a month: $20.15 for the EC2 instance and $3.00 for the 30GB EBS attached storage.
3. Create and configure your EC2 instance so that only your machine can ssh into the instance.
    - Setup your security group to only allow inbound from SSh from your computer.  And outbound traffic can go anywhere.
    - Setup IAM so that the ec2 instance can access S3
    - Try to get a spot instance if possible.
    - In the UserData Section add the following commands: `aws s3 cp s3://uj-scratch/udacity/ec2-setup.sh . && chmod +x ec2-setup.sh`
4. Once the ec2 instance has been launched
    - Run the ec2-setup.sh script on the remote ec2 instance
    - On local machine use VSCode to SSH into your EC2 instance (should have that set up in the `~/.ssh/config` file)
    - Install the Pylance extension on VSCode for the remote ec2 instance
    - Open Jupyter notebook: landmark.ipynb
    - Set the Juypter noteobook interpreter to the pytorch 3.9.13 interpreter
5. When you are done, before closing the ssh connection, remember to save changed project files to S3

---

### Filtering non-informative images? 2

Looking at the images (e.g., Taj_Mahal), I found some pictures may have very little or no information.  So for example, all pixels are the same.  I should probably filter this out because it will reduce noise.  This should be true for all images (train and test).  So is there a way for me to create a cleaner that remove images with little variation?  Or will the neural network handle this?  

The network would possible figure this out.  But I could reduce processing time by removing?  I guess this is not a priority.
