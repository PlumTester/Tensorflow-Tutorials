# https://www.tensorflow.org/tutorials/keras/regression

# aim to predict output of continuous value, eg price / probability
# predicts fuel efficiency of late 1970s and early 1980s automobiles
# includes cylinders, displacement, horsepower, weight, etc

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

print(tf.__version__)
print("")

# download the dataset
dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print(dataset_path)

column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']

dataset = pd.read_csv(dataset_path, names=column_names, na_values = "?", comment='\t', sep=" ", skipinitialspace=True)
# print(dataset.tail())

print("")
print("CLEANING THE DATA...")
# Clean the data
# print(dataset.isna().sum())
dataset = dataset.dropna()

# map origin variable from enumeration to strings
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3:'Japan'})
# print(dataset.tail())

# replaced origin variable with dummy variables for each country
dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
print(dataset.tail())

# split into train, test
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# inspect the data
sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
# plt.show()
plt.close()

# overall statistics
train_stats = train_dataset.describe().transpose()
print()
print(train_stats)

# split the features from labels
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

print("--")
print(train_dataset.head())
print("--")

# normalise the data
# without normalisation, model dependent on choice of units, makes training more difficult, etc
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
    # return (x - train_stats.drop(['MPG'], axis=0)['mean']) / train_stats.drop(['MPG'], axis=0)['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

normed_train_data = normed_train_data.drop(['MPG'], axis=1)
normed_test_data = normed_test_data.drop(['MPG'], axis=1)

print()
print("BUILDING THE MODEL...")
# sequential model, two fully connected layers, output layer returning single, continuous value
# fully connected layers both have 64 nodes
# input layer train_dataset keyd (cylinders, displacement, horsepower, weight, acceleration, model year, europe, japan, usa) connects to first layer of 64
def build_model():
    model = keras.Sequential([ \
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]), \
        layers.Dense(64, activation='relu'), \
        layers.Dense(1) \
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse', \
                    optimizer=optimizer, \
                    metrics=['mae', 'mse'])
    
    return model

# call the build model function
model = build_model()

# inspect model
print(model.summary())

# model test run - batch of 10 examples
# example_batch = normed_train_data.drop(['MPG'], axis=1)[:10]
example_batch = normed_train_data[:10]
print(example_batch.head())

# NOTE make sure that the target variable is not in the data here (MPG is NaaN) when normalised for some reason?

# put batch through the model
example_result = model.predict(example_batch)

# look to see if model seems to be working
# should have result 10 large, (array of 10 errors)
print(example_result)

print()
print("TRAINING THE MODEL...")

EPOCHS = 1000

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

"""
# The normalisation brought MPG back, remember to drop it before training
history = model.fit(normed_train_data, \
                        train_labels, \
                        epochs=EPOCHS, \
                        validation_split = 0.2, \
                        verbose=0, \
                        callbacks=[tfdocs.modeling.EpochDots()] \
                        )

# visualises models training progress
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

plotter.plot({'Basic': history}, metric = "mae")
plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')
# plt.show()
plt.close()

plotter.plot({'Basic': history}, metric = "mse")
plt.ylim([0, 20])
plt.ylabel('MSE [MPG^2]')
# plt.show()
plt.close()

"""

# if loss through validation phase doesn't improve mse
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

# The normalisation brought MPG back, remember to drop it before training
early_history = model.fit(normed_train_data, \
                        train_labels, \
                        epochs=EPOCHS, \
                        validation_split = 0.2, \
                        verbose=0, \
                        callbacks=[early_stop, tfdocs.modeling.EpochDots()] \
                        )

plotter.plot({'Early Stopping': early_history}, metric = 'mae')
plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')
# plt.show()
plt.close()

plotter.plot({'Early Stopping': early_history}, metric = "mse")
plt.ylim([0, 20])
plt.ylabel('MSE [MPG^2]')
# plt.show()
plt.close()

# TESTING
print()
print("TESTING...")

# Tells how well the model can be expected to predict (~1.8-2.1 MPG)
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
print('Testing set Mean Abs Error: {:5.2f}MPG'.format(mae))

# make predictions
test_predictions = model.predict(normed_test_data).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values[MPG]')
plt.ylabel("Predictions [MPG]")
lims=[0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
# plt.show()
plt.close()

# take a look at the error distribution
# look for this to be gaussian
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()
plt.close()


# NOTES
#       MSE - common loss function used for regression problems
#       MAE - evaluation metric for regression problem 
#       when numeric features have values with different ranges, features should be normalised
#       if not much training data, prefer a small network with few hidden layers to avoid overfitting
#       early stopping is useful to prevent overfitting