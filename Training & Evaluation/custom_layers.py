# https://www.tensorflow.org/tutorials/customization/custom_layers#layers_common_sets_of_useful_operations

import tensorflow as tf




def main():

    basic_custom_layer()
    
    return


# general layer notes
# inspect layer variables
    # print(layer.variables)
    # gives tensor of kernel (weights) and bias for a given layer

# access variables weights and bias via variable
    # print(layer.kernel)
    # print(layer.bias)

# implementing custom layers
    # extend tf.keras.layers.Layer
    # implement...
        # __init__  : where you can do all input-independent intialization
        # build     : where you know the shapes of input tensors and can do the rest of init
        # call      : where you do forward computation

    # note can create variables in __init___
        # advantage to build
        # it enables late variable creation based on the shape of inputs the layer will operate on 
        # creating variables in __init__ means shapes required need to be explicitly specified

def basic_custom_layer():
    class MyDenseLayer(tf.keras.layers.Layer):
        def __init__(self, num_outputs):
            super(MyDenseLayer, self).__init__()
            self.num_outputs = num_outputs

        def build(self, input_shape):
            self.kernel = self.add_weight("kernel",
                                        shape=[int(input_shape[-1]),
                                                self.num_outputs])

        def call(self, input):
            return tf.matmul(input, self.kernel)

    layer = MyDenseLayer(10)

    _ = layer(tf.zeros([10, 5])) # Calling the layer `.builds` it.

    print([var.name for var in layer.trainable_variables])





if __name__ == '__main__':
    main()