# https://www.tensorflow.org/guide/keras/custom_callback#introduction

import tensorflow as tf
from tensorflow import keras

import numpy as np

# overview of callback methods
#   global
    #   on_(train|test|predict)_begin(self, logs=None)
    #       called beginning of fit, evalute, or predict
    #   on_(train|test|predict)_end(self, logs=None)
    #       called end of fit, evaluate, predict

#   batch level
    #   on batch, begin, batch end

# epoch level
    #   on epoch begin, epoch end


def main():

    # Load example MNIST data and pre-process it
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

    # Limit the data to 1000 samples
    x_train = x_train[:1000]
    y_train = y_train[:1000]
    x_test = x_test[:1000]
    y_test = y_test[:1000]

    model = get_model()

    callbacks_list = [ \
                        AccuracyCallback(), \
                        # LossAndErrorPrintingCallback(), \
                        # EarlyStoppingAtMinLoss(), \
                        # get_CSV_Logger(), \
                    ]

    model.fit( \
        x_train, \
        y_train, \
        batch_size=128, \
        epochs=1, \
        verbose=0, \
        validation_split=0.5, \
        callbacks=callbacks_list, \
    )

    res = model.evaluate(x_test, y_test, batch_size = 128, verbose=0, callbacks=callbacks_list)

    res = model.predict(x_test, batch_size=128, callbacks=callbacks_list)

    return


class AccuracyCallback(keras.callbacks.Callback):


    def on_epoch_end(self, epoch, logs=None):
        print('EPOCH OVER BRUH')




# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/CSVLogger
# CSV Logger
def get_CSV_Logger():

    callback = tf.keras.callbacks.CSVLogger('log.csv', separator=',', append=False)

    return callback


# callbacks have access to the model associated with current round of training / evaluation / inference
    # self.model
# usage of self.model
#   interrupt training at self.model.stop_training = True
#   mutate hyperparameters of optimizer (eg self.model.optimizer.learning_rate)
#   save model at period intervals
#   record output of model.pridect() on a few test samples at the end of an epoch (sanity check during training)
#   Extract visualisations of intermediate features at the end of an epoch to monitor what the model is learning over time


# examples of callback applications

class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, patience=0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))





# loss and error printing callback
class LossAndErrorPrintingCallback(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    def on_test_batch_end(self, batch, logs=None):
        print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    def on_epoch_end(self, epoch, logs=None):
        print(
            "The average loss for epoch {} is {:7.2f} "
            "and mean absolute error is {:7.2f}.".format(
                epoch, logs["loss"], logs["mean_absolute_error"]
            )
        )


# BASIC Custom callback
# custom call backgs heavily use a logs dict
# the logs dict contains loss value and all metrics at the end of a batch or epoch
class CustomCallback(keras.callbacks.Callback):

    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start testing; got log keys: {}".format(keys))

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop testing; got log keys: {}".format(keys))

    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start predicting; got log keys: {}".format(keys))

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop predicting; got log keys: {}".format(keys))

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))



    



# Define the Keras model to add callbacks to
def get_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(1, input_dim=784))
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.1),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )
    return model



if __name__ == "__main__":
    main()
