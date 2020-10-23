# https://www.tensorflow.org/guide/keras/train_and_evaluate#setup

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os
import datetime

def main():

    inputs = keras.Input(shape=(784,), name="digits")
    x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(10, activation="softmax", name="predictions")(x)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess the data (these are NumPy arrays)
    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255

    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    # Reserve 10,000 samples for validation
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]

    # tensorboard_visualization_during_training(x_train, y_train)

    # custom_validation_metric_attempt()

    return




def PLOT_MODEL_EXAMPLE():
    model = get_compiled_model()
    keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)


# def custom_validation_callback_attempt():

#     # saves list of per-battch loss values during training
#     class MetricsCallback(keras.callbacks.Callback):

#         def on_train_begin(self, logs):
#             self.per_batch_losses = []

#         def on_batch_end(self, batch, logs):
#             self.per_batch_losses.append(logs.get('loss'))


#     model = get_compiled_model()

#     callbacks = MetricsCallback()


#     return

# command line to open
    # tensorboard --logdir=/logs/scalar

# https://www.tensorflow.org/tensorboard/scalars_and_keras

def tensorboard_visualization_during_training(x_train, y_train):
    
    model = get_compiled_model()

    logdir = os.path.join('logs', 'scalars', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=logdir,
            histogram_freq=0,  # How often to log histogram visualizations
            embeddings_freq=0,  # How often to log embedding visualizations
            update_freq="epoch",
        )  # How often to write logs (default: once per epoch)
    ]

    model.fit(
        x_train, y_train, epochs=30, batch_size=64, callbacks=callbacks, validation_split=0.2
    )



# saving model checkpoint
def save_chkpt_callback(x_train, y_train):

    # Prepare a directory to store all the checkpoints.
    checkpoint_dir = "./ckpt"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    model = make_or_restore_model()

    simple_callback = [
        keras.callbacks.ModelCheckpoint(
            # Path where to save the model
            # The two parameters below mean that we will overwrite
            # the current checkpoint if and only if
            # the `val_loss` score has improved.
            # The saved model name will include the current epoch.
            filepath="mymodel_{epoch}",
            save_best_only=True,  # Only save a model if `val_loss` has improved.
            monitor="val_loss",
            verbose=1,
        )
    ] 

    save_restore_callbacks = [
        keras.callbacks.ModelCheckpoint(
            # This callback saves a SavedModel every 100 batches.
            # We include the training loss in the saved model name.
            filepath=checkpoint_dir + "/ckpt-loss={loss:.2f}", 
            save_freq=100
        )
    ]

    model.fit(
        x_train, y_train, epochs=2, batch_size=64, callbacks=save_restore_callbacks, validation_split=0.2
    )


# def load_chkpt():

def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.

    # Prepare a directory to store all the checkpoints.
    checkpoint_dir = "./ckpt"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return get_compiled_model()




# write custom callback - extend base class
# saves list of per batch loss values during training
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.per_batch_losses = []

    def on_batch_end(self, batch, logs):
        self.per_batch_losses.append(logs.get("loss"))

# built in callbacks
#   Model Checkpoint - periodically save the model
#   early stoppping - stop when trianing no longer improving validation metrics
#   tensorboard - periodically write model logs that can be visualized in TensorBoard
#   CSV Logger - streams loss and metrics data to csv file

# using callbakcs
# called at different points during training
    # eg start of epoch, end of batch, end of epoch
# used to implement behaviours
    # validation at different points during training (beyond built in once per epoch validation)
    # checkpointing model at regular interval / when it receives a certain accuracy threshold
    # changing learning rate when plateauing
    # doing fine-tuning of top layers when training seems to be plateauing
# callbacks are passed as list to fit()

def early_stopping_callback(x_train, y_train):
    model = get_compiled_model()

    callbacks = [
        keras.callbacks.EarlyStopping(
            # Stop training when `val_loss` is no longer improving
            monitor="val_loss",
            # "no longer improving" being defined as "no better than 1e-2 less"
            min_delta=1e-2,
            # "no longer improving" being further defined as "for at least 2 epochs"
            patience=2,
            verbose=1,
        )
    ]
    model.fit(
        x_train,
        y_train,
        epochs=20,
        batch_size=64,
        callbacks=callbacks,
        validation_split=0.2,
    )




def training_from_tf_datasets(x_train, y_train, x_test, y_test):

    model = get_compiled_model()

    n_examples = len(x_train)
    slice_i = int(n_examples * 0.2)

    x_val = x_train[slice_i:n_examples]
    y_val = y_train[slice_i:n_examples]

    x_train = x_train[0:slice_i]
    y_train = y_train[0:slice_i]

    # Prepare the training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

    # Prepare the validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(64)

    # Now we get a test dataset.
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(64)

    # Since the dataset already takes care of batching,
    # we don't pass a `batch_size` argument.
    model.fit(train_dataset, epochs=3, validation_data=val_dataset)

    # You can also evaluate or predict on a dataset.
    print("Evaluate")
    result = model.evaluate(test_dataset)
    dict(zip(model.metrics_names, result))



# automatically setting apart a validation holdout set
def basic_matric_logging_layer(x_train, y_train):
    inputs = keras.Input(shape=(784,), name="digits")
    x = layers.Dense(64, activation="relu", name="dense_1")(inputs)

    # Insert std logging as a layer.
    x = MetricLoggingLayer()(x)

    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(10, name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    model.fit(x_train, y_train, batch_size=64, validation_split=0.2, epochs=1)


# add a metrics logging layer
class MetricLoggingLayer(layers.Layer):
    def call(self, inputs):
        # The `aggregation` argument defines
        # how to aggregate the per-batch values
        # over each epoch:
        # in this case we simply average them.
        self.add_metric(
            keras.backend.std(inputs), name="std_of_activation", aggregation="mean"
        )
        return inputs  # Pass-through layer.


def basic_custom_metric(x_train, y_train):

    model = get_uncompiled_model()
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[CategoricalTruePositives()],
    )

    model.fit(x_train, y_train, batch_size=64, epochs=3)


# custom METRICS
# create by subclassing metrics.Metric class
# implement 4 methods - __init__, update_state, result, reset_states
class CategoricalTruePositives(keras.metrics.Metric):
    def __init__(self, name='categorical_true_positives', **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='ctp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1,1))
        values = tf.cast(y_true, 'int32') == tf.cast(y_pred, 'int32')
        values = tf.cast(values, 'float32')

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)

        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_states(self):
        # state reset at start of each spoch
        self.true_positives.assign(0.0)




# custom_losses_additional_parameters
    # subclass losses.Loss class and implement the init and call methods
class CustomMSE(keras.losses.Loss):
    def __init__(self, regularization_factor=0.1, name="custom_mse"):
        super().__init__(name=name)
        self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):
        mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
        reg = tf.math.reduce_mean(tf.square(0.5 - y_pred))
        return mse + reg * self.regularization_factor


def custom_losses_additional_parameters(x_train, y_train):
    model = get_uncompiled_model()
    model.compile(optimizer=keras.optimizers.Adam(), loss=CustomMSE())

    y_train_one_hot = tf.one_hot(y_train, depth=10)
    model.fit(x_train, y_train_one_hot, batch_size=64, epochs=1)


# custom loss function 
def basic_custom_losses(x_train, y_train):
    model = get_uncompiled_model()
    model.compile(optimizer=keras.optimizers.Adam(), loss=custom_mean_squared_error)

    # We need to one-hot encode the labels to use MSE
    y_train_one_hot = tf.one_hot(y_train, depth=10)
    model.fit(x_train, y_train_one_hot, batch_size=64, epochs=1)



# creates function that accepts inputs y_true and y_pred
def custom_mean_squared_error(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))


def get_uncompiled_model():
    inputs = keras.Input(shape=(784,), name="digits")
    x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(
        optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model



def basic_run_through():

    inputs = keras.Input(shape=(784,), name="digits")
    x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(10, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)


    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess the data (these are NumPy arrays)
    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255

    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    # Reserve 10,000 samples for validation
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]


    model.compile(
        optimizer=keras.optimizers.RMSprop(),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.SparseCategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    print("Fit model on training data")
    history = model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=2,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(x_val, y_val),
    )

    print(history.history)


    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=128)
    print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print("Generate predictions for 3 samples")
    predictions = model.predict(x_test[:3])
    print("predictions shape:", predictions.shape)


if __name__ == "__main__":
    main()