# https://www.tensorflow.org/guide/keras/custom_layers_and_models#setup

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

# SHOWS HOW to print custom training bits at the bottom in end to end

def main():

    end_to_end_example()




# layer object
# encapsulates state (weights) and transformation from inputs to outputs (call, layer's forward pass)
    # weights will have varaiables w (weights) and b (biases)
    # use a layer by calling it on some tensor inputs (eg python function)
# can add weight to a layer .add_weight() method

def add_weight_example():
    class Linear(keras.layers.Layer):
        
        def __init__(self, units=32, input_dim=32):
            super(Linear, self).__init__()
            self.w = self.add_weight(
                shape=(input_dim, units), initializer="random_normal", trainable=True
            )
            self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

        def call(self, inputs):
            return tf.matmul(inputs, self.w) + self.b


    x = tf.ones((2, 2))
    print(x)

    linear_layer = Linear(4, 2)
    y = linear_layer(x)
    print(y)


def non_trainable_weights_example():
    class ComputeSum(keras.layers.Layer):
        def __init__(self, input_dim):
            super(ComputeSum, self).__init__()
            self.total = tf.Variable(initial_value=tf.zeros((input_dim,)), trainable=False)

        def call(self, inputs):
            self.total.assign_add(tf.reduce_sum(inputs, axis=0))
            return self.total


    x = tf.ones((2, 2))
    print(x)
    my_sum = ComputeSum(2)
    y = my_sum(x)
    print(y)
    y = my_sum(x)
    print(y)


    # input is [[1,1],[1,1]]
    # not important to understand reduce_sum function, just some operation in call
    # output is [2, 2] and [4, 4]


def add_metric_to_layer():
    class LogisticEndpoint(keras.layers.Layer):
        def __init__(self, name=None):
            super(LogisticEndpoint, self).__init__(name=name)
            self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
            self.accuracy_fn = keras.metrics.BinaryAccuracy()

        def call(self, targets, logits, sample_weights=None):
            # Compute the training-time loss value and add it
            # to the layer using `self.add_loss()`.
            loss = self.loss_fn(targets, logits, sample_weights)
            self.add_loss(loss)

            # Log accuracy as a metric and add it
            # to the layer using `self.add_metric()`.
            acc = self.accuracy_fn(targets, logits, sample_weights)
            self.add_metric(acc, name="accuracy")

            # Return the inference-time prediction tensor (for `.predict()`).
            return tf.nn.softmax(logits)

    inputs = keras.Input(shape=(3,), name="inputs")
    
    targets = keras.Input(shape=(10,), name="targets")
    logits = keras.layers.Dense(10)(inputs)
    
    predictions = LogisticEndpoint(name="predictions")(logits, targets)

    model = keras.Model(inputs=[inputs, targets], outputs=predictions)
    model.compile(optimizer="adam")

    data = {
        "inputs": np.random.random((3, 3)),
        "targets": np.random.random((3, 10)),
    }

    model.fit(data)



def privileged_training_call_method():
    class CustomDropout(keras.layers.Layer):
        def __init__(self, rate, **kwargs):
            super(CustomDropout, self).__init__(**kwargs)
            self.rate = rate

    # NOTE EXPOSED training argument here allows for training vs validation / testing specific call on a layer
        def call(self, inputs, training=None):
            if training:
                return tf.nn.dropout(inputs, rate=self.rate)
            return inputs


# implement variational auto-encoder on MNIST digits
def end_to_end_example():

    class Sampling(layers.Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    
    class Encoder(layers.Layer):
        """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

        def __init__(self, latent_dim=32, intermediate_dim=64, name="encoder", **kwargs):
            super(Encoder, self).__init__(name=name, **kwargs)
            self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
            self.dense_mean = layers.Dense(latent_dim)
            self.dense_log_var = layers.Dense(latent_dim)
            self.sampling = Sampling()

        def call(self, inputs):
            x = self.dense_proj(inputs)
            z_mean = self.dense_mean(x)
            z_log_var = self.dense_log_var(x)
            z = self.sampling((z_mean, z_log_var))
            return z_mean, z_log_var, z

    class Decoder(layers.Layer):
        """Converts z, the encoded digit vector, back into a readable digit."""

        def __init__(self, original_dim, intermediate_dim=64, name="decoder", **kwargs):
            super(Decoder, self).__init__(name=name, **kwargs)
            self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
            self.dense_output = layers.Dense(original_dim, activation="sigmoid")

        def call(self, inputs):
            x = self.dense_proj(inputs)
            return self.dense_output(x)


    class VariationalAutoEncoder(keras.Model):
        """Combines the encoder and decoder into an end-to-end model for training."""

        def __init__(
            self,
            original_dim,
            intermediate_dim=64,
            latent_dim=32,
            name="autoencoder",
            **kwargs
        ):
            super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
            self.original_dim = original_dim
            self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
            self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

        def call(self, inputs):
            z_mean, z_log_var, z = self.encoder(inputs)
            reconstructed = self.decoder(z)
            # Add KL divergence regularization loss.
            kl_loss = -0.5 * tf.reduce_mean(
                z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
            )
            self.add_loss(kl_loss)
            return reconstructed

    original_dim = 784
    vae = VariationalAutoEncoder(original_dim, 64, 32)

    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype("float32") / 255

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    mse_loss_fn = tf.keras.losses.MeanSquaredError()

    loss_metric = tf.keras.metrics.Mean()
    

    
    custom_training_loop = True

    if not custom_training_loop:
        vae.compile(optimizer, loss=mse_loss_fn)
        vae.fit(train_dataset, epochs=2, batch_size=64)
    
    else:

        # CUSTOM TRAINING LOOP
        epochs = 2

        # Iterate over epochs. 
        for epoch in range(epochs):
            print("Start of epoch %d" % (epoch,))

            # Iterate over the batches of the dataset.
            for step, x_batch_train in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    reconstructed = vae(x_batch_train)
                    # Compute reconstruction loss
                    loss = mse_loss_fn(x_batch_train, reconstructed)
                    loss += sum(vae.losses)  # Add KLD regularization loss

                grads = tape.gradient(loss, vae.trainable_weights)
                optimizer.apply_gradients(zip(grads, vae.trainable_weights))

                loss_metric(loss)

                if step % 100 == 0:
                    print("step %d: mean loss = %.4f" % (step, loss_metric.result()))


if __name__ == '__main__':
    main()