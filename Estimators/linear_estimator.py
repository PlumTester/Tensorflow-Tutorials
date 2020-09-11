# https://www.tensorflow.org/tutorials/estimator/linear

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import clear_output

from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

# ROC import
from sklearn.metrics import roc_curve



def main():

    dftrain, dfeval, y_train, y_eval, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, feature_columns = load_data()

    train_input_fn = make_input_fn(dftrain, y_train)
    eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

    # # # inspect dataset
    # ds = make_input_fn(dftrain, y_train, batch_size=10)()

    # # inspect items in batch
    # for feature_batch, label_batch in ds.take(1):
    #     print('some feature keys:', list(feature_batch.keys()))
    #     print()
    #     print('a batch of class', feature_batch['class'].numpy())
    #     print()
    #     print('a batch of labels:', label_batch.numpy())
    #     print()

    #     # inspect result of specific feature using DenseFeatures layer
    #     # only accepts a dense tensor
    #     # to inspect categorical, you need to transform that to an indicator first
    #     age_column = feature_columns[7]
    #     print(tf.keras.layers.DenseFeatures([age_column])(feature_batch).numpy())
    #     print()


    linear_est = basic_estimator_training(feature_columns, train_input_fn, eval_input_fn)


    # derived_ft_estimator_training(feature_columns, train_input_fn, eval_input_fn)

    # make predictions on passenger from evaluation set
    # models optimized to make predictions on a batch (or collection) of examples at once
    #   eval_input_fn was defined using entire evaluation set
    pred_dicts = list(linear_est.predict(eval_input_fn))
    probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
    
    # shows predicted probability for each passenger in the evaluation dataset
    probs.plot(kind='hist', bins=20, title='predicted probabilities')
    # plt.show()
    plt.close()

    # look at the ROC of results - (receiver operating characteristics)
    # gives better idea of the tradeoff between true positive rate and false positive rate
    fpr, tpr, _ = roc_curve(y_eval, probs)
    plt.plot(fpr, tpr)

    plt.title('ROC curve')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.xlim(0,)
    plt.ylim(0,)
    # plt.show()
    plt.close()

    return


def load_data():

    # load titanic data
    dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
    dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
    y_train = dftrain.pop('survived')
    y_eval = dfeval.pop('survived')

    # # explore the data
    # print(dftrain.head())
    # print()
    # print(dftrain.describe())
    # print()

    # # 627 passengers in train
    # # 264 passengers in evaluation
    # print(dftrain.shape[0], dfeval.shape[0])

    
    dftrain.age.hist(bins=20)
    # plt.show()
    plt.close()

    # approx 2x male passengers vs female passengers
    dftrain.sex.value_counts().plot(kind='barh')
    # plt.show()
    plt.close()

    # major in third class
    dftrain['class'].value_counts().plot(kind='barh')
    # plt.show()
    plt.close()

    # females have much higher chance of surviving versus males - predictive feature for the model
    pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
    # plt.show()
    plt.close()

    # FEATURE ENGINEERING FOR MODEL

    CATEGORICAL_COLUMNS = ['sex', \
                            'n_siblings_spouses', \
                            'parch', \
                            'class', \
                            'deck', \
                            'embark_town', \
                            'alone']

    NUMERIC_COLUMNS = ['age', \
                            'fare']
    

    feature_columns = []

    for feature_name in CATEGORICAL_COLUMNS:

        # get a set of range of feature values
        vocabulary = dftrain[feature_name].unique()
        # define range of categorical values for feature
        feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

    for feature_name in NUMERIC_COLUMNS:
        # define numerical dtype
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))


    return dftrain, dfeval, y_train, y_eval, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, feature_columns


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):

    def input_function():
        # note instantiate TF Dataset from pandas df by dict transformation
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))

        if shuffle:
            ds = ds.shuffle(1000)

        ds = ds.batch(batch_size).repeat(num_epochs)

        return ds

    return input_function


def basic_estimator_training(feature_columns, train_input_fn, eval_input_fn):

    # train the model - single command using estimator API
    linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
    linear_est.train(train_input_fn)
    result = linear_est.evaluate(eval_input_fn)

    clear_output()

    print(result)

    return linear_est


def derived_ft_estimator_training(feature_columns, train_input_fn, eval_input_fn):

    # create a derived numerical feature
    age_x_gender = tf.feature_column.crossed_column(['age', 'sex'], hash_bucket_size=100)
    derived_feature_columns = [age_x_gender]

    # train the model - single command using estimator API
    linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns + derived_feature_columns)
    linear_est.train(train_input_fn)
    result = linear_est.evaluate(eval_input_fn)

    clear_output()

    print(result)

    return linear_est


if __name__ == "__main__":
    main()