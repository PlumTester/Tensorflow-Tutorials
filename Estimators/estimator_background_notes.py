#   https://www.tensorflow.org/guide/estimator

import tensorflow as tf

import pandas as pd
import numpy as pd

import os


# tf.estimator.Estimator Notes
#       wraps a model which is specified by a model_fn
#           given inputs and other parameters
#       returns operations necessary to perform training, evaluation, or predictions
#       Ouputs written to model_dir or a subdirectory there of - eg checkpoints, event files
#       Arguments
#           config - passed to tf.estimator.RunConfig object (model_fn has parameter named config)
#                    contains information about execution environment
#                    not passing config means defaults useful for local execution are used
#           params - contains hyperparameters (passed to model_fn)
#                    estimator object only passes params along, does not inspect it
#                    structure of params is entirely up to the developer
#       Estimator's methods - cannot be overridden
#           subclasses should override model_fun to configure base class


####    tf.estimator.Estimator(model_fun, model_dir=None, config=None, params=None, warm_start_from=None)


def main():
    
    warm_start_estimator()

    return


# ARGS for Estimator

# model_fn ARG
#   features - first item returned from input_fn passed to train, evaluate, and predict
#               should be a single tf.Tensor or dict of same
#   labels -   second item returned from input_fn passed to train, evaluate, and predict
#               should be single tf.Tensor or dict of same (for multi-head models).
#               if mode is tf.estimator.ModeKeys.PREDICT, labels=None will be passed
#               if model_fn's signature does not accept mode, the model_fn must still be able to handle labels=None
#   mode -     (Optional), specifies if training, eavluation, or prediction
#               allows to configure estimators from hyper parameter tunig
#   config -   (Optional), estimator.RunConfig object
#               will receive what is passed to estimator as its config parameter or default value
#               allows setting up things in your model_fn based on configurations such as num_ps_replicas or model_dir
#   Returns tf.estimator.EstimatorSpec
def model_fn(features, labels, mode, config):
    pass

# model_dir ARG
#   directory to save model parameters, graphs, etc
#   used to load checkpoints from directory into an estimator to continue training a previously saved model
#   if PathLike object, path will be resolved
#   if None, model_dir in config will be used if set
#   if both set, they must be the same
#   if both are None, temporary directory will be used

# config ARG
#   estimator.RunConfig configuration object

# params ARG
#   dict of hyperparamteres that will be passed into model_fn
#   Keys are names of parameters, values are basic python types

# warm_start_from ARG
#   (Optional) string filepath to a checkpoint or SavedModel to warm-start from or a tf.estimator.WarmStartSettings object to fully configure warm-starting
#   If None, only TRAINABLE variables are warm-started
#   If string filepath is provided instead of tf.estimator.WarmStartSettings, the ALL variables are warm-started
#       It is assumed that vocabularies and tf.Tensor names are unchanged

# EXAMPLE warm-start an estimator
# this means that you can intiialise weights in the model from a predefined checkpoint
# can be done for all weights in the model, only the input layer, only TRAINABLE variables, etc
def warm_start_estimator():
    estimator = tf.estimator.DNNClassifier( \
        feature_columns=[categorical_feature_a_emb, categorical_feature_b_emb], \
        hidden_units=[1024,512,256], \
        warm_start_from="/path/to/checkpoint/dir" )

# Errors
#   ValueError - model_fn parameters don't match params
#   ValueError - if this is called via a subclass and if class overrides a member of Estimator



if __name__ == "__main__":
    main()