# https://www.tensorflow.org/tutorials/images/cnn

# simple CNN classification of CIFAR images

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


def main():

    train_images, train_labels, test_images, test_labels, class_names = load_verify_data()

    # create convolutional base
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # input shape parameter
    #   takes tensors of shape (height, width, color_channels) - ignoring batch size


    # image shape (32, 32, 3)

    # first cnn layer - 30x30 filter outputs * 3 = 900 * 3 = 2352 * 32 nodes = 75264

    # add Dense layers on top
    #   feed output tensor from CNN base (4, 4, 64) into dense layers to perform classification
    # first required 4, 4, 64 to be flattened
    # then put into a 1D dense layer
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    

    # structure
    # 32 node conv2d with 3x3 kernel - (3, 30, 30, 32) 
    # max pooling with 2x2 kernel - (32, 15, 15, 32) 
    # 64 node conv2d with 3x3 kernel - (3, 13, 13, 64)
    # max pooling with 2x2 kernel - (3, 6, 6, 64)
    # 64 node conv2d with 3x3 kernel - (3, 4, 4, 64) 
    # flatten 4x4x64 to 1024
    # dense 64 node
    # output 10 node
    print(model.summary())
    
    # compile and train
    model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=10, 
                        validation_data=(test_images, test_labels))


    # evaluate
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()
    plt.close()

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print(test_acc)

    return


def load_verify_data():

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # verify data
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        # The CIFAR labels happen to be arrays, 
        # which is why you need the extra index
        plt.xlabel(class_names[train_labels[i][0]])
    # plt.show()
    plt.close()

    return train_images, train_labels, test_images, test_labels, class_names



if __name__ == "__main__":
    main()

