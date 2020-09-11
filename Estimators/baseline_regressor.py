# https://www.tensorflow.org/api_docs/python/tf/estimator/BaselineRegressor#args

import tensorflow as tf

from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

import os

# Notes
#   Baseline Regressor - a regressor that can establish a simple baseline
#       ignores feature values
#       learns to predict the average value of each label
#       

def main():
 
    # build baseline regressor
    regressor = tf.estimator.BaselineRegressor()

    # Fit model
    regressor.train(input_fn=input_fn_train)

    # Evaluates cross entropy between the test and train labels
    loss = regressor.evaluate(input_fn=input_fn_eval)['loss']

    # predict outputs the mean value seen during training
    predictions = regressor.predict(new_samples)
    


# Input builders
def input_fn_train():
    # returns tf.data.Dataset of (x, y) tuple where y represents label's class index
    pass

def input_fn_eval():
    # Returns tf.data.Dataset of (x, y) tuple where y represents label's class index
    pass




if __name__ == "__main__":
    main()



