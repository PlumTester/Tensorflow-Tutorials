# https://www.tensorflow.org/guide/migrate#estimators


import tensorflow as tf
import tensorflow_datasets as tfds

# tfds contains utilities for laoding predefined datasets as tf.data.Dataset objects



def main():

    training_with_estimators()
    
    return


def input_fn():

    # load MNIST dataset
    datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
    mnist_train, mnist_test = datasets['train'], datasets['test']

    BUFFER_SIZE = 10000
    BATCH_SIZE = 64

    train_data = mnist_train.map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return train_data.repeat()


def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255

    return image, label[..., tf.newaxis]


def training_with_estimators():

    # define train and eval specs
    STEPS_PER_EPOCH = 5
    NUM_EPOCHS = 5

    # define train, eval specs
    train_spec = tf.estimator.TrainSpec(input_fn=input_fn, max_steps = STEPS_PER_EPOCH * NUM_EPOCHS)
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn, steps=STEPS_PER_EPOCH)

    model = make_keras_model_definition()

    model.compile(optimizer='adam', \
                    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), \
                    metrics=['accuracy'])

    estimator = tf.keras.estimator.model_to_estimator(keras_model = model)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    return


def make_keras_model_definition():
    return tf.keras.Sequential([
                                    tf.keras.layers.Conv2D(32, 3, activation='relu',
                                                        kernel_regularizer=tf.keras.regularizers.l2(0.02),
                                                        input_shape=(28, 28, 1)),
                                    tf.keras.layers.MaxPooling2D(),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dropout(0.1),
                                    tf.keras.layers.Dense(64, activation='relu'),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(10)
                            ])



if __name__ == "__main__":
    main()