# https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit#introduction

import tensorflow as tf
from tensorflow import keras
import numpy as np

def main():

    custom_evaluation_step_example()


# what happens in fit

# write own training loop - use GradientTape to take control of all details

# higher level than GradientTape - override training step function of model Class
# then call fit normally, but running personal learning algorithm


def simple_example():

    # subclass keras.Model
    # override method train_step(self, data)
    # return dictionary mapping metric names (including the loss) to their current value


    # in body of train_step method - implement regular training updates similar to what you're familiar with
    # compute loss via self.compiled_loss - wraps loss(es) functions that were passed to compile
    # compiled_metrics.update_state to update the state of the metrics that were passed in compile
    # query results from self.metrics at the end to retrieve current value

    class CustomModel(keras.Model):
        def train_step(self, data):
            # Unpack the data. Its structure depends on your model and
            # on what you pass to `fit()`.
            x, y = data

            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)  # Forward pass
                # Compute the loss value
                # (the loss function is configured in `compile()`)
                loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            # Update metrics (includes the metric that tracks the loss)
            self.compiled_metrics.update_state(y, y_pred)
            # Return a dict mapping metric names to current value
            return {m.name: m.result() for m in self.metrics}

    # Construct and compile an instance of CustomModel
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model = CustomModel(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # Just use `fit` as usual
    x = np.random.random((1000, 32))
    y = np.random.random((1000, 1))
    model.fit(x, y, epochs=3)

    return


# manually do loss and metrics within train_step
def lower_level_example():

    # create Metric instances to track loss and MAE score
    # implement train-step, update metrics (update_state()), query metrics (result())
    # need to call reset_states on metrics per epoch
    loss_tracker = keras.metrics.Mean(name="loss")
    mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

    class CustomModel(keras.Model):
        def train_step(self, data):
            x, y = data

            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)  # Forward pass
                # Compute our own loss
                loss = keras.losses.mean_squared_error(y, y_pred)

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Compute our own metrics
            loss_tracker.update_state(loss)
            mae_metric.update_state(y, y_pred)
            return {"loss": loss_tracker.result(), "mae": mae_metric.result()}

        @property
        def metrics(self):
            # We list our `Metric` objects here so that `reset_states()` can be
            # called automatically at the start of each epoch
            # or at the start of `evaluate()`.
            # If you don't implement this property, you have to call
            # `reset_states()` yourself at the time of your choosing.
            return [loss_tracker, mae_metric]


    # Construct an instance of CustomModel
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model = CustomModel(inputs, outputs)

    # We don't passs a loss or metrics here.
    model.compile(optimizer="adam")

    # Just use `fit` as usual -- you can use callbacks, etc.
    x = np.random.random((1000, 32))
    y = np.random.random((1000, 1))
    model.fit(x, y, epochs=5)


def sample_weight_example():

    # supporting fit arugments sample weight / class weight
    # unpack sample_w from data argument
    # pass it to compiled_loss & compiled_metrics

    class CustomModel(keras.Model):
        def train_step(self, data):
            # Unpack the data. Its structure depends on your model and
            # on what you pass to `fit()`.
            if len(data) == 3:
                x, y, sample_weight = data
            else:
                x, y = data

            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)  # Forward pass
                # Compute the loss value.
                # The loss function is configured in `compile()`.
                loss = self.compiled_loss(
                    y,
                    y_pred,
                    sample_weight=sample_weight,
                    regularization_losses=self.losses,
                )

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Update the metrics.
            # Metrics are configured in `compile()`.
            self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)

            # Return a dict mapping metric names to current value.
            # Note that it will include the loss (tracked in self.metrics).
            return {m.name: m.result() for m in self.metrics}


    # Construct and compile an instance of CustomModel
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model = CustomModel(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # You can now use sample_weight argument
    x = np.random.random((1000, 32))
    y = np.random.random((1000, 1))
    sw = np.random.random((1000, 1))
    model.fit(x, y, sample_weight=sw, epochs=3)


def custom_evaluation_step_example():
    # override test_step in exact same way
    class CustomModel(keras.Model):
        def test_step(self, data):
            # Unpack the data
            x, y = data
            # Compute predictions
            y_pred = self(x, training=False)
            # Updates the metrics tracking the loss
            self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            # Update the metrics.
            self.compiled_metrics.update_state(y, y_pred)
            # Return a dict mapping metric names to current value.
            # Note that it will include the loss (tracked in self.metrics).
            return {m.name: m.result() for m in self.metrics}


    # Construct an instance of CustomModel
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model = CustomModel(inputs, outputs)
    model.compile(loss="mse", metrics=["mae"])

    # Evaluate with our custom test_step
    x = np.random.random((1000, 32))
    y = np.random.random((1000, 1))
    model.evaluate(x, y)



if __name__ == '__main__':
    main()