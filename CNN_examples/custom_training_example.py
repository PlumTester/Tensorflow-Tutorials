# https://www.tensorflow.org/guide/migrate#estimators


import tensorflow as tf
import tensorflow_datasets as tfds

# tfds contains utilities for laoding predefined datasets as tf.data.Dataset objects



def main():

    dataset_training_example()
    
    return


def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255

    return image, label



def dataset_training_example():

    # load MNIST dataset
    datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
    mnist_train, mnist_test = datasets['train'], datasets['test']

    # print(mnist_train.element_spec)

    BUFFER_SIZE = 10
    BATCH_SIZE = 64
    NUM_EPOCHS = 5

    # prepare for training
    #   rescale each image
    #   shuffle the order of the examples
    #   collect batches of images and labels

    # train data maps : https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map
    #   maps scale function across all elements of the dataset

    # shuffle: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle
    #   randomly shuffles elements of this dataset
    #   fills buffer with buffer_size of elements, randomly samples elements from this buffer (replacing selected elements with new elements)
    #   perfect shuffling - buffer_size >= full size of dataset is required
    #   if choose buffer_size as subset - once datapoint randomly chosen within buffer, next point outside buffer will replace that position in the buffer

    # batch: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch
    #   combines consecutive elements of dataset into batches
    #       [0 ,1, 2, 3, 4, 5, 6, 7] .batch(3) ==> [[0,1,2], [3,4,5], [6,7]]
    #   NOTE - can choose to drop_remainder
    train_data = mnist_train.map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_data = mnist_test.map(scale).batch(BATCH_SIZE)

    # trim dataset to only return 5 batches
    # I think steps per epoch means only train each epoch on 5 batches of 64 buffer_sized images
    #   hence - 5 epochs of 5 batches of 64 * 10 images
    STEPS_PER_EPOCH = 5
    train_data = train_data.take(STEPS_PER_EPOCH)
    test_data = test_data.take(STEPS_PER_EPOCH)
    # NOTE ^ prefetch before was take
    #       take:       creates a dataset with at most STEPS_PER_EPOCH elements from this dataset
    #       prefetch:   creates dataset that prefetches elements from this dataset
    #           input pipelines should end with a call to prefetch
    #           allows later elements to be prepared while current element is being processed

    image_batch, label_batch = next(iter(train_data))

    # keras_training_loop(train_data, test_data, NUM_EPOCHS)

    custom_training_loop(image_batch, label_batch, train_data, test_data, NUM_EPOCHS)


# Keras training Loops
    #   default fit, evaluate, and predict
    #   advantages
    #       accept numpy arrays, python generators, and tf.data.Datasets
    #       apply regularization and activation losses automatically
    #       support tf.distirbute for multi-device training
    #       support arbitrary callables as losses and metrics
    #       support callbackes like TensorBoard and custom callbacks
    #       performant, automatically using Tensorflow graphs
def keras_training_loop(train_data, test_data, NUM_EPOCHS):\
    
    # CNN model
    # 1 Conv layer
    #       filter = 32         ; dimensionality of output space (number of output filters in convolution)
    #       kernel_size = 3     ; square kernel of 3x3
    #       activation relu, l2 regularization
    #       input_shape = (28, 28, 1)
    #           ? each batch is 64 images of 28,28,1 ?
    # 1 max pooling 2D layer
    # flatten the 
    # dropout layer
    # fully connected layer - 64 nodes
    model = tf.keras.Sequential([
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

    # NOTE 
    #   conv output is 26x26 * 32 nodes
    #   max pooling out is 13x13 * 32 nodes
    #   flatten is 13 x 13 x 32 nodes, flat 5408 nodes
    #   dropout 5048
    #   dense 64
    #   not sure what batch normalisation does....
    #   Dense output layer of 10 for mnist dataset prediction
    print(model.summary())

    model.compile(optimizer='adam', \
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), \
                    metrics=['accuracy'])

    model.fit(train_data, epochs=NUM_EPOCHS)
    loss, acc = model.evaluate(test_data)

    print("Loss {}, Accuracy {}".format(loss, acc))

    return



# if need more control outside the default model compile, fit, and evaluate
#   utilises Model.train_on_batch method
#   gives user control of the outer loop
#   test_on_batch and evaluate are also used to check performance during training
def standard_training_loop(image_batch, label_batch, train_data, test_data, NUM_EPOCHS):

    # train and test _on_batch both return loss and metrics for batch by default
    
    # Model
    model = tf.keras.Sequential([
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

    print(model.summary())
    # Compile full model without custom layers

    model.compile(optimizer='adam', \
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), \
                    metrics=['accuracy'])

    # custom training loops
    for epoch in range(NUM_EPOCHS):

        # reset metric accumulators
        model.reset_metrics()

        for image_batch, label_batch in train_data:
            result = model.train_on_batch(image_batch, label_batch)
            metrics_names = model.metrics_names

            print("train: ", \
                "{}: {:.3f}".format(metrics_names[0], result[0]), \
                "{}: {:.3f}".format(metrics_names[1], result[1]))

    # custom testing loops
    for image_batch, label_batch in test_data:
        # returns accumulated metrics
        result = model.test_on_batch(image_batch, label_batch, reset_metrics=False)

    metrics_names = model.metrics_names
    print("\neval: ", \
        "{}: {:.3f}".format(metrics_names[0], result[0]), \
        "{}: {:.3f}".format(metrics_names[1], result[1]))

    return


# this is the above training loop with customisations
#   iteratoes over Pyhton generator or tf.data.Dataset to get batches of examples
#   uses tf.GradientTape to collect gradients
#   uses tf.keras.optimizers to apply weight updates to the model's variables
#
def custom_training_loop(image_batch, label_batch, train_data, test_data, NUM_EPOCHS):

    # train and test _on_batch both return loss and metrics for batch by default
    
    # Model
    model = tf.keras.Sequential([
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

    print(model.summary())
    # Compile full model without custom layers

    optimizer = tf.keras.optimizers.Adam(0.001)

    # loss object is callable and expectes y_true, y_pred arguments
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # metric object is callable - calling state with new observations returns new result of metric
    # metric object has methods...
    #   update_state    -   add new observations
    #   result          -   get current result of the metric, given the observed values
    #   reset_states    -   clear all observations
    # TF has automatic control of depenecies, initialises metric's variables for you
    loss_metric = tf.metrics.Mean(name='train_loss')
    accuracy_metric = tf.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            regularization_loss = tf.math.add_n(model.losses)
            pred_loss = loss_fn(labels, predictions)
            total_loss = pred_loss + regularization_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # update metrics
        loss_metric.update_state(total_loss)
        accuracy_metric.update_state(labels, predictions)

    for epoch in range(NUM_EPOCHS):

        # reset the metrics
        loss_metric.reset_states()
        accuracy_metric.reset_states()

        for inputs, labels in train_data:
            train_step(inputs, labels)

        # Get metric results
        mean_loss = loss_metric.result()
        mean_accuracy = accuracy_metric.result()

        print('Epoch: ', epoch)
        print('  loss:     {:.3f}'.format(mean_loss))
        print('  accuracy: {:.3f}'.format(mean_accuracy))



    # # loss object is callable and expectes y_true, y_pred arguments
    # cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # print(cce([[1, 0]], [[-1.0, 3.0]]).numpy())

    return

