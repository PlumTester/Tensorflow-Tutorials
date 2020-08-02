# https://www.tensorflow.org/tutorials/keras/overfit_and_underfit

# accuracy of model peaks after training for a number of epochs - starts decreasing thereafter (overfit)
# accuracy of model has room for improvement (underfit)

# overfitting - model learns patterns from training data which don't generalize
# best solution - use more compelte training data, cover full range of inputs
# other techniques - regularization: place constraints on quantity / type o finfo model stores
#                                    forces network to memorize a small number of paterns (most prominent patterns)

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import regularizers

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

from IPython import display
from matplotlib import pyplot as plt

import numpy as np

import pathlib
import shutil
import tempfile

print(tf.__version__)
print()

logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)

# Higgs dataset - contains 11 000 000 examples
# each example has 28 features and binary class label
gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')
FEATURES = 28

# used to read csv records directly from gzip file with no intermediate decompression step
# returns a list of scalars for each record
ds = tf.data.experimental.CsvDataset(gz, [float(),]*(FEATURES+1), compression_type="GZIP")

# repacks list of scalars into a (feature_vector, label) pair
def pack_row(*row):
    label = row[0]
    features = tf.stack(row[1:], 1)
    return features, label

# make a new dataset that repacks rows via batches
# 1 batch = 10 000 examples
# apply pack_row function to each batch, split batches back into individual records
packed_ds = ds.batch(10000).map(pack_row).unbatch()

# have a look at one of the feature, label sets
for features, label in packed_ds.batch(1000).take(1):
    print(features[0])
    plt.hist(features.numpy().flatten(), bins = 101)
    # plt.show()
    plt.close()

# first 1000 samples for validation, next 10,000 for training
N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE
# ^ double slash rounds to the nearest whole number (floor division) I think it truncates

# Dataset cche ensures loader doesn't need to re-read data from file on each epoch
validate_ds = packed_ds.take(N_VALIDATION).cache()
train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()

# These datsets return individual examples
# batch method: create batches of appropriate size for training
print(train_ds)


# before batching, remember to shuffle and repeat training set (is this done)
validate_ds = validate_ds.batch(BATCH_SIZE)
train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)


# simple approach - prevent overfitting, minimise number of learnable parameters
# learnable parameters - a model's "capacity"
# model with greater parameters, more 'memorization capacity'
#       learn dictionary-like mapping between training and targets, little generalisation power
# limited memorization resources - minimise loss by learning compressed representations which have more predictive power
#       if mdoel too small, difficulty fitting to training data
#       balance between too much capacity, too little capacity
#       best approach to determine capcity - experiment with series of different architectures

# start with few layers and parameters
# increase size of layers, add noew layers until diminishing returns on validation loss

# TRAINING procedure
# many models train better if you gradually reduce the learning rate during training
#       use optimizers.shcedules to reduce the learning rate over time

# defined schedule which hyperbolically decreases learning rate to 1/2 of the base rate at 1000 epoch, 1/3 of base rate at 2000 epochs, etc
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay( \
    0.001, \
    decay_steps = STEPS_PER_EPOCH * 1000, \
    decay_rate = 1, \
    staircase = False \
    )

def get_optimizer():
    return tf.keras.optimizers.Adam(lr_schedule)

# visualisation of the LR over time (epochs)
step = np.linspace(0, 100000)
lr = lr_schedule(step)
plt.figure(figsize = (8,6))
plt.plot(step/STEPS_PER_EPOCH, lr)
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
_ = plt.ylabel('Learning Rate')
# plt.show()
plt.close()

# tfdocs.EpochDots simpl prints a . for each epoch and full set of metrics every 100 epochs
# this reduces the logging noise
# EarlyStopping callback avoids long / unecessary training times
#       monitors val_binary_crossentropy, not val_loss - IMPORTANT LATER
# TensorBoard generates TensorBoard logs for training

# each model will use the same set of callbacks
def get_callback(name):
    return [ \
        tfdocs.modeling.EpochDots(), \
        tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200), \
        tf.keras.callbacks.TensorBoard(logdir/name) \
        ]

# each model will use the same compile and fit
def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
    if optimizer is None:
        optimizer = get_optimizer()
    
    model.compile(optimizer=optimizer, \
                    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True), \
                    metrics = [ \
                                tf.keras.losses.BinaryCrossentropy(from_logits=True, name='binary_crossentropy'), \
                                'accuracy'
                            ]    
                )
    
    print(model.summary())

    history = model.fit( \
                            train_ds, \
                            steps_per_epoch = STEPS_PER_EPOCH, \
                            epochs = max_epochs, \
                            validation_data = validate_ds, \
                            callbacks = get_callback(name), \
                            verbose = 0 \
                        )

    return history

plotter = tfdocs.plots.HistoryPlotter(metric = 'binary_crossentropy', smoothing_std=10)


# Tiny model - VERSION 1
# input layer (28), hidden layer (16), output layer (1)
tiny_model = tf.keras.Sequential([ \
                                    layers.Dense(16, activation='elu', input_shape=(FEATURES,)), \
                                    layers.Dense(1) \
                                ])

# story different model results
size_histories = {}

# run the model, save in histories
size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny')

# view model training process
plotter.plot(size_histories)
plt.ylim([0.5, 0.7])
# plt.show()
plt.close()

"""

# progressively train larger models
# SMALL model
# input model 28, hidden 1 (16 nodes), hidden 2 (16 nodes), output (1 node)
small_model = tf.keras.Sequential([ \
                                    layers.Dense(16, activation='elu', input_shape=(FEATURES,)), \
                                    layers.Dense(16, activation='elu'), \
                                    layers.Dense(1) \
                                ])

size_histories['Small'] = compile_and_fit(small_model, 'sizes/Small')

# MEDIUM model
# input layer 28, hidden 1 (64 nodes), hidden 2 (64 nodes), hidden 3 (64 nodes), output (1 node)
medium_model = tf.keras.Sequential([ \
                                    layers.Dense(64, activation='elu', input_shape=(FEATURES, )), \
                                    layers.Dense(64, activation='elu'), \
                                    layers.Dense(64, activation='elu'), \
                                    layers.Dense(1)
                                ])

size_histories['Medium'] =compile_and_fit(medium_model, 'sizes/Medium')

# LARGE model
# input layer (28), hidden 1 (512), hidden 2 (512), hidden 3 (512), output (1)
large_model = tf.keras.Sequential([ \
                                    layers.Dense(512, activation='elu', input_shape=(FEATURES,)), \
                                    layers.Dense(512, activation='elu'), \
                                    layers.Dense(512, activation='elu'), \
                                    layers.Dense(512, activation='elu'), \
                                    layers.Dense(1) \
                                ])

size_histories['Large'] = compile_and_fit(large_model, 'sizes/Large')

plotter.plot(size_histories)
a = plt.xscale('log')
plt.xlim([5, max(plt.xlim())])
plt.ylim([0.5, 0.7])
plt.xlabel("Epochs [Log Scale]")
# plt.show()
plt.close()

"""

# copy training logs from Tiny to use as a baseline for comparison

shutil.rmtree(logdir/'regularizers/Tiny', ignore_errors=True)
shutil.copytree(logdir/'sizes/Tiny', logdir/'regularizers/Tiny')

regularizer_histories = {}
regularizer_histories['Tiny'] = size_histories['Tiny']

# Strategies to prevent overfitting

# Weight regularization
# ---------------------

# put constraints on complexity of a network by forcing its weights only to take small values
# makes distribution of weight values more 'regular'
# done by adding to the loss function of the network + a cost associated with having large weights
#       L1 regularization - cost added is proportional to absolute value of weights coefficients (ie the 'L1 norm' of the weights)
#                           pushes weights towards exactly zero - encouraging a sparse model
#       L2 regularization - cost added is proportional to the square of the value of the weights coefficient (ie the 'L2 norm' of the weights)
#                           called weighted decay, mathetmatically the same as L2 regularization
#                           penalises weight parameters without making them sparse, penalty goes to zero for small weights (this is why L2 is more common)

# keras adds weight regularization by passing weight regularizer instances to layers as keyword arguments
l2_model = tf.keras.Sequential([ \
                                    layers.Dense(512, \
                                                    activation='elu', \
                                                    kernel_regularizer = regularizers.l2(0.001), \
                                                    input_shape = (FEATURES,) \
                                                ), \
                                    layers.Dense(512, \
                                                    activation='elu', \
                                                    kernel_regularizer=regularizers.l2(0.001) \
                                                ), \
                                    layers.Dense(512, \
                                                    activation='elu', \
                                                    kernel_regularizer=regularizers.l2(0.001) \
                                                ), \
                                    layers.Dense(512, \
                                                    activation='elu', \
                                                    kernel_regularizer=regularizers.l2(0.001) \
                                                ), \
                                    layers.Dense(1)
                                ])

regularizer_histories['l2']  = compile_and_fit(l2_model, 'regularizers/l2')

plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])
plt.show()
plt.close()













