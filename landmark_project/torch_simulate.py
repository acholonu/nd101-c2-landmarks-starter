from typing import Tuple, Union
import pandas as pd


class Layer(object):
    def __init__(
                self,
                layer_name:str, #Layer you are going to apply
                input_depth:int,
                filter_size:Tuple[int], #kernel_size
                input_dim:Union[int,Tuple[int]],
                requested_num_filters:int,
                stride:int =1, # Move 1 over
                padding:int = 0,
                is_conv_layer:bool = True,
            ) -> None:
        
        # Characteristics of Layer
        self.name = layer_name
        self.is_conv_layer = is_conv_layer

        # Inputs into the Layer
        self.input_depth = input_depth
        self.input_dim = None
        self.stride = stride
        self.padding = padding
        self.filter_size = filter_size
        self.num_filters = requested_num_filters

        # Outputs of Layer
        self.dim:Tuple[int] = None
        self.num_weights:int

        if isinstance(input_dim,int):
            self.input_dim = (input_dim,input_dim)
        elif isinstance(input_dim,Tuple[int]):
            self.input_layer_dim  = input_dim
        else:
            ValueError("Layer dimension must be an int or Tuple")

    # Test case: use quiz results
    def determine_padding(self)->int:
        recommended_padding = self.dim[0]%self.filter_size
        return(recommended_padding)

    def set_padding(self,new_padding):
        self.padding = new_padding

    def calc_layer_shape(self)->Tuple:
        """Calculate spatial dimension of a convolutional layer"""
        depth = self.num_filters

        if self.is_conv_layer:
            width = (self.input_dim[0] - self.filter_size[0]+2*self.padding)/self.stride + 1
            height = (self.input_dim[1] - self.filter_size[1]+2*self.padding)/self.stride + 1
        else:
            width = self.input_dim[0]/self.filter_size[0]
            height = self.input_dim[1]/self.filter_size[1]
        result = (width,height,depth)
        return(result)

    def calc_num_weights(self)->int:
        """Calculate the total number weights that will be estimated in the convolutional layer.
           
           AHA Moment: Remember your convolutional layer is really trying to find the filter weights to use
           for the kernel size you determine and for the number of filters you request!  These are the best
           filter to find differences than can lead to identification.
        """
        if self.is_conv_layer == False:
            return None

        if self.num_weights is None:
            num_weights_per_filter = self.filter_size[0] * self.filter_size[1] * self.input_depth
            self.num_weights = num_weights_per_filter * self.num_filters # This is also the depth of the final layer
        
        return(self.num_weights)

    def calc_num_parameters(self)->int:
        """Calculate the number of parameters the algorithm is trying to locate. Remember,
        Since there is one bias term per filter, the convolutional layer has K (where K = num_filters) 
        biases.
        """
        if self.is_conv_layer == False:
            return None
        bias = self.num_filters
        self.num_parameters = bias + self.calc_num_parameters()
        return(self.num_parameters)

    def get_layer_output(self)->dict:
        results = {}
        results['shape'] = self.calc_layer_shape()
        if self.is_conv_layer:
            results['num_weights'] = self.calc_num_weights() 
            results['num_parameters'] = self.calc_num_parameters() # num_weight + Bias terms
        return(results)


class NetSimulate(object):
    def __init__(self) -> None:
        """Constructor"""
        self.layers = {}

    def add_layer(self, layer:Layer):
        # Check if layer by that name has already been added
        self.layers[layer.name] = layer

def summary():
    print("testing")

if __name__ == "__main__":
    summary()