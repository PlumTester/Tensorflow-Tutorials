# https://www.tensorflow.org/tutorials/estimator/boosted_trees

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve

from IPython.display import clear_output

import tensorflow as tf


def main():

    dftrain, dfeval, y_train, y_eval, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, feature_columns = load_data()

    
    # use entire batch since this is such a small dataset
    num_examples = len(y_train)

    train_input_fn = make_input_fn(dftrain, y_train, num_examples)
    eval_input_fn = make_input_fn(dfeval, y_eval, num_examples, num_epochs=1, shuffle=False)

    # let's first train a linear classifier (logistic regression model)
    # best practice to start with simpler model to establish a benchmark
    linear_est = tf.estimator.LinearClassifier(feature_columns)

    # train linear model
    linear_est.train(train_input_fn, max_steps = 100)

    # Evaluation
    result = linear_est.evaluate(eval_input_fn)
    linear_result_series = pd.Series(data=list(result.values()), index=list(result.keys()), name='linear', dtype='float32')
    clear_output()

    # print(pd.Series(result))

    # Boosted Trees Model
    # since data fits in memory - use entire dataset per layer (it will be faster)
    # above one batch is defined as the entire dataset
    n_batches = 1
    bt_est = tf.estimator.BoostedTreesClassifier(feature_columns=feature_columns, n_batches_per_layer=n_batches)

    # model will stop training once the specified number of trees is built (NOT basied on # of steps)
    bt_est.train(train_input_fn, max_steps=100)

    # Eval
    result = bt_est.evaluate(eval_input_fn)
    bt_result_series = pd.Series(data=list(result.values()), index=list(result.keys()), name='boost trees', dtype='float32')
    clear_output()

    # combine results into one dataframe
    result_df = pd.DataFrame(data=[linear_result_series, bt_result_series]).transpose()
    print(result_df)

    # make predictions from boosted trees
    pred_dicts = list(bt_est.predict(eval_input_fn))
    probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
    
    # plot histogram of probabilities
    probs.plot(kind='hist', bins=20, title='predicted probabilities')
    # plt.show()
    plt.close()

    # plot roc curve
    fpr, tpr, _ = roc_curve(y_eval, probs)
    plt.plot(fpr, tpr)
    plt.title('ROC curve')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.xlim(0,)
    plt.ylim(0,)
    plt.show()
    plt.close()

    return
    


def make_input_fn(data_df, label_df, num_examples, num_epochs=None, shuffle=True):

    def input_function():
        # note instantiate TF Dataset from pandas df by dict transformation
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))

        if shuffle:
            ds = ds.shuffle(num_examples)

        # for training - cycle through dataset as many times as needed (n_epochs=None)
        ds = ds.repeat(num_epochs)

        # in memory, training doesn't use batching
        ds = ds.batch(num_examples)

        return ds

    return input_function



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
    
    def one_hot_cat_column(feature_name, vocab):
        return tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab))
        
    feature_columns = []

    for feature_name in CATEGORICAL_COLUMNS:
        # get a set of range of feature values
        vocabulary = dftrain[feature_name].unique()
        # define range of categorical values for feature
        feature_columns.append(one_hot_cat_column(feature_name, vocabulary))


    for feature_name in NUMERIC_COLUMNS:
        # define numerical dtype
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))


    # # see an example of what one_hot_cat_column does
    # # tuple of numerical dummy variables basically
    # example = dict(dftrain.head(1))
    # class_feature_column = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('class', ('First', 'Second', 'Third')))
    # print('Feature value: "{}"'.format(example['class'].iloc[0]))
    # print('One-hot encoded: ', tf.keras.layers.DenseFeatures([class_feature_column])(example).numpy())

    # # can also view all feature column transformations together
    # print(tf.keras.layers.DenseFeatures(feature_columns)(example).numpy())

    return dftrain, dfeval, y_train, y_eval, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, feature_columns



if __name__ == "__main__":
    main()