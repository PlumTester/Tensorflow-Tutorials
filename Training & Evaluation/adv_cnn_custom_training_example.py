import os
import matplotlib.pyplot as plt
import tensorflow as tf

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))


# optimizer - applies gradients to model's variables to minimize the loss function
# think of loss function as c urved surface
# # 

def main():

    training_loop()

    return



def training_loop():

    train_ds = generate_train_dataset()

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(3)
    ])

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    # define loss function
        # loss - measure of how off a model's predictions are from the desired label - how bad is the model
            # we need to minimise this value
        # model will calcualte loss using SpareCategoricalCrossentropy
    def loss(model, x, y, training):
        # training = training is needed only if there are layers with different behaviour during training versus inference (eg Dropout)
        y_ = model(x, training=training)

        return loss_object(y_true=y, y_pred=y_)

    # define gradient function
    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = loss(model, inputs, targets, training=True)

        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    
    # iterate each epoch
    # within epoch, iterate each example (grab featuers x, label y)
    # using features, make prediction and compare it with label
        # measure inaccuracy of prediction and use that to calculate the model's loss and gradients
    # use an optimizer to update the model's variables
    # keep track of some stats for visualisation
    # repeat for each epoch

    # keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    num_epochs = 201

    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalCrossentropy()

        # training loop - batches of 32
        for x, y in train_ds:
            # optimize the model
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # track progress
            epoch_loss_avg.update_state(loss_value)

            # Compare predicted label to actual label 
            # training=True is needed only if there are layers with different
            # behaviour during training versus inference (eg Dropout)
            epoch_accuracy.update_state(y, model(x, training=True))

        # end epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))

    # evalute effectiveness
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss_results)

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(train_accuracy_results)
    # plt.show()
    plt.close()

    test_ds = generate_test_dataset()

    test_accuracy = tf.keras.metrics.Accuracy()

    for (x, y) in test_ds:

        # training = False, needed only if there are layers with different behaviour (eg Dropout)
        logits = model(x, training=False)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)

    print("Test set accuracy {:.3%}".format(test_accuracy.result()))

    tf.stack([y,prediction],axis=1)



def generate_test_dataset():

    batch_size = 32
    # column order in CSV file
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

    test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"

    test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                     origin=test_url)
    
    test_dataset = tf.data.experimental.make_csv_dataset(
        test_fp,
        batch_size,
        column_names=column_names,
        label_name='species',
        num_epochs=1,
        shuffle=False)

    def pack_features_vector(features, labels):
        """Pack the features into a single array."""
        features = tf.stack(list(features.values()), axis=1)
        return features, labels

    test_dataset = test_dataset.map(pack_features_vector)

    return test_dataset


def get_simple_model(train_ds):

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(3)
    ])

    features, labels = next(iter(train_ds))

    # predictions = model(features)
    # print(predictions[:5])

    # print(tf.nn.softmax(predictions[:5]))

    # print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
    # print("    Labels: {}".format(labels))

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # define loss function
        # loss - measure of how off a model's predictions are from the desired label - how bad is the model
            # we need to minimise this value
        # model will calcualte loss using SpareCategoricalCrossentropy
    def loss(model, x, y, training):
        # training = training is needed only if there are layers with different behaviour during training versus inference (eg Dropout)
        y_ = model(x, training=training)

        return loss_object(y_true=y, y_pred=y_)

    # define gradient function
    def grad(model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = loss(model, inputs, targets, training=True)

        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    l = loss(model, features, labels, training=False)
    print("Loss test: {}".format(l))

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    loss_value, grads(model, features, labels)

    print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                            loss_value.numpy()))

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
                                            loss(model, features, labels, training=True).numpy()))

    return model







def generate_train_dataset():

    train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

    train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                            origin=train_dataset_url)

    print("Local copy of the dataset file: {}".format(train_dataset_fp))

    # column order in CSV file
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

    feature_names = column_names[:-1]
    label_name = column_names[-1]

    print("Features: {}".format(feature_names))
    print("Label: {}".format(label_name))

    class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

    batch_size = 32

    train_dataset = tf.data.experimental.make_csv_dataset(
        train_dataset_fp,
        batch_size,
        column_names=column_names,
        label_name=label_name,
        num_epochs=1)

    # repackage geatures dictionary intoa  single array with shape (batch_size, num_features)
    # tf.stack takes values from a list of tensors and creates a combined tensor at the specified dimension
    def pack_features_vector(features, labels):
        """Pack the features into a single array."""
        features = tf.stack(list(features.values()), axis=1)
        return features, labels

    train_dataset = train_dataset.map(pack_features_vector)

    # test
    features, labels = next(iter(train_dataset))
    # print(features[:5])

    return train_dataset



if __name__ == '__main__':
    main()