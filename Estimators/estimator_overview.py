# https://www.tensorflow.org/guide/estimator#pre-made_estimators

import tensorflow as tf
import tensorflow_datasets as tfds


# Premade Estimators
#   let you experiment with different model architectures by making only minimal code changes
# Structure
#   1. write one or more dataset importing functions
    #   create on function to import training set, another to import test set
    #   each dataset importing function must return two objects
    #   1 - dictionary in which keys are feature names and the values are Tensors (or SparseTensors) containing the corresponding feature data
    #   2 - Tensor containing one or more labels

#   2. define feature columns
    #   tf.feature_column identifies a feature name, its type, and any input pre-processing
    #   a lambda can be specified to invoke a scaling of the raw data

#   3. instantiate the relevant pre-made Estimator
    #   https://www.tensorflow.org/tutorials/estimator/linear

#   4. call training, evaluation, or inference method
    # all estimaors provide a trian method


# don't think this will compile

def main():

    population, crime_rate, median_education = example_feature_columns()

    estimator = sample_instantiation(population, crime_rate, median_education)

    # don't know what format my_training_set would be
    my_training_set = None
    sample_train_eval_inference(estimator, my_training_set)

    return


# step 1
# illustrates basic skeleton for an input function
def input_fn(dataset):
    
    # manipulate dataset, extracting the feature dict and the label

    feature_dict = {}
    label = {}

    # data guide - https://www.tensorflow.org/guide/data

    return feature_dict, label


# step 2
# illustrates basic feature column definition
def example_feature_columns():

    # can hold integer or floating-point data
    population = tf.feature_column.numeric_column('population')
    crime_rate = tf.feature_column.numeric_column('crime_rate')

    # random init
    global_education_mean = 10
    
    # specifies lambda for scaling
    median_education = tf.feature_column.numeric_column('median_education', normalizer_fn=lambda x: x- global_education_mean)

    # features columns tutorial - https://www.tensorflow.org/tutorials/keras/feature_columns

    return population, crime_rate, median_education


# step 3
def sample_instantiation(population, crime_rate, median_education):
    estimator = tf.estimator.LinearClassifier( feature_columns = [ population, crime_rate, median_education ] )
    return estimator
    

# step 4
def sample_train_eval_inference(estimator, my_training_set):
    # input_fun is the function created in Step 1
    estimator.train(input_fn=my_training_set, steps=2000)

    return

if __name__ == "__main__":
    main()