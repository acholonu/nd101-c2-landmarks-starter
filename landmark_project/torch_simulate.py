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
        self.input_dim = input_dim
        self.stride = stride
        self.padding = padding
        self.filter_size = filter_size
        self.num_filters = requested_num_filters

        # Outputs of Layer
        self.dim:Tuple[int] = None
        self.num_weights:int = None
        self.num_parameters:int = None

        if isinstance(input_dim,int):
            self.input_dim = (input_dim,input_dim)
        elif isinstance(input_dim,Tuple):
            self.input_layer_dim  = input_dim
        else:
            ValueError("Layer dimension must be an int or Tuple")

    # Test case: use quiz results
    def determine_padding(self)->int:
        """Determine padding need based on filter size and image size

        TODO: I need to make sure that I test the calculation below
        """
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
        if self.num_weights is None:
            self.num_parameters = bias + self.calc_num_weights()
        else:
            self.num_parameters = bias + self.num_weights

        return(self.num_parameters)

    def get_layer_output(self)->dict:
        results = {}
        results['shape'] = self.calc_layer_shape()
        if self.is_conv_layer:
            results['num_weights'] = self.calc_num_weights() 
            results['num_parameters'] = self.calc_num_parameters() # num_weight + Bias terms
        else:
            results['num_weights'] = None 
            results['num_parameters'] = None
            
        return(results)


class NetSimulate(object):
    def __init__(self) -> None:
        """Constructor"""
        self.layers = []

    def add_layer(self, layer:Layer):
        # Check if layer by that name has already been added
        self.layers.append(layer)

    def print_summary(self)->pd.DataFrame:
        summary_layers = []

        for layer in self.layers:
            result = layer.get_layer_output()
            row = [result['shape'],result['num_weights'],result['num_parameters']]
            summary_layers.append(row)

        df = pd.DataFrame(summary_layers, columns=["dimension","num_weights","num_parameters"])
        print(df)
        return(df)

def summary():
    print("testing")
    conv1 = Layer(
            layer_name="conv1",
            input_depth = 3,
            filter_size = (3,3),
            input_dim = (256,256),
            requested_num_filters = 16,
            stride = 1,
            padding = 1,
            is_conv_layer= True,
        )

    conv2 = Layer(
            layer_name="conv2",
            input_depth = 16,
            filter_size = (3,3),
            input_dim = (256,256),
            requested_num_filters = 32,
            stride = 1,
            padding = 1,
            is_conv_layer= True,
        )

    max1 = Layer(
            layer_name="max1",
            input_depth = 32,
            filter_size = (2,2),
            input_dim = (256,256),
            requested_num_filters = 0,
            stride = 2,
            padding = 0,
            is_conv_layer= False,
        )

    conv3 = Layer(
            layer_name="conv3",
            input_depth = 32,
            filter_size = (3,3),
            input_dim = (128,128),
            requested_num_filters = 64,
            stride = 1,
            padding = 1,
            is_conv_layer= True,
        )

    max2 = Layer(
            layer_name="max2",
            input_depth = 64,
            filter_size = (2,2),
            input_dim = (256,256),
            requested_num_filters = 0,
            stride = 2,
            padding = 0,
            is_conv_layer= False,
        )

    model = NetSimulate()
    model.add_layer(conv1)
    model.add_layer(conv2)
    model.add_layer(max1)
    model.add_layer(conv3)
    model.add_layer(max2)
    df = model.print_summary()
    df.head()


if __name__ == "__main__":
    summary()