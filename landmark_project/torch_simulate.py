from typing import Tuple, Union
import pandas as pd


class Layer(object):
    def __init__(
                self,
                layer_name:str, #Layer you are going to apply
                layer_type:str,
                input_depth:int,
                filter_size:Tuple[int], #kernel_size
                input_dim:Union[int,Tuple[int]],
                stride:int =1, # Move 1 over
                padding:int = None
            ) -> None:
        
        # Characteristics of Layer
        self.name = layer_name
        self.type = layer_type

        # Inputs into the Layer
        self.input_depth = input_depth
        self.input_dim = None
        self.stride = stride
        self.padding = padding
        self.filter_size = filter_size

        # Outputs of Layer
        self.depth:int
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
        return(None)

    def set_padding(self,new_padding):
        self.padding = new_padding

    def calc_spatial_dimension(self)->Tuple:
        """Calculate spatial dimension of a convolutional layer"""
        return(None)

    def calc_num_weights(self)->Tuple:
        """Calculate the resulting dimension of convolutional layer"""
        return(None)

    def calc_depth_size(self)->Tuple:
        "Calculate the resulting depth of a convolutional layer"
        return(None)

    def get_layer_output(self)->dict:

        results = {}
        results['depth'] = self.calc_depth_size()
        results['dim'] = self.calc_spatial_dimension()
        results['num_weights'] = self.calc_num_weights() # or number of parameters
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