# https://www.tensorflow.org/tutorials/keras/classification

# classify images of clothing (eg sneakers and shirts)

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
print("------")
print()

# import fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

### description of loading the data ###

# four NumPy arrays
#       train_images, train_labels - training set: the data the model uses to learn
#       test_images, test_labels - testing set: model is tested against these examples

# images are 28x28 NumPy arrays, pixel values raning from 0, 255

# labels are array of integers, ranging from 0 to 9
# integer corresponds to class of clothing the image represents
#       0, t-shirt / top
#       1, trouser
#       2, pullover
#       3, dress
#       4, coat
#       5, sandal
#       6, shirt
#       7, sneaker
#       8, bag
#       9, ankle boot
# each image is mapped to a single label

# declare class names (not included in dataset) - for plotting later
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', \
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# first explore the data
print("EXPLORING DATA...")
# 60,000 images in training set, 28x28 pixel format
print(train_images.shape)

# 60,000 labels in training set
print(len(train_labels))

# each label is integer between 0, 9
print(train_labels[:5])

# 10,000 images in test set, 28x28 pixel format
print(test_images.shape)

# 10,000 images labels
print(len(test_labels))
print("")

# second, preprocess data
print("PREPROCESSING DATA...")

# inspect first image in training set, see pixel values fall in range 0-255
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
# plt.show()
plt.close()

# these 0-255 values must be scaled to 0-1 (normalised)
# training and testing set must be preprocessed in the same way
train_images = train_images / 255.0
test_images = test_images / 255.0

# display first 25 images from training set
# display class name below each image
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
# plt.show()
plt.close()

print("")


# third, build the model
print("BUILDING MODEL...")

# NN build of layers
# layers extract representations from data fed into them
# these representations are meaningful for the problem at hand
# DL consists of chaining simple layers

model = keras.Sequential([ \
    keras.layers.Flatten(input_shape=(28,28)), \
    keras.layers.Dense(128, activation='relu'), \
    keras.layers.Dense(10) \
])

### model description ###
# input layer for 28, 28 pixels
#       transforms format of images from two dimensional array (28, 28) to one dimensional array of 28* 28 = 784 pixels
#       'unstacking' rows of pixels in image, lining them up
#       no parameters to learn, only reformats data
# fully connected layer to 128 neurons
# fully connected layer to 10 neurons - one for each class
#       returns logit arrays with length of 10
#       each node contains a score that indicates the current image belongs to one of the 10 classes


# BEFORE compiling the model - define loss function, optimizer, and metrics
# loss function - measures how accurate model is during training. Minimse function to 'steer' the model in the right direction
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Optimizer - how the model is updated based on the data it sees and its loss function
# adam will be used

# Metrics - used to monitor training and testing steps
# accuracy will be used = fraction of images that are correctly classified

model.compile(optimizer='adam', \
    loss=loss_fn, \
    metrics=['accuracy'])

# training the model requires four steps
# 1. feed training data into the model (images, labels)
# 2. model learns associate images and labels
# 3. ask model to make predictions about test set - test_images array
# 4. verify predictions match labels from test_labels array

# step 1 - feed the model
model.fit(train_images, train_labels, epochs=10)

# step 2 - tf.keras learns the model across the 10 epochs

# step 3 - ask model to make predictions
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

# step 4 - verify predictions
# compare how the model performs on the dataset
print('\nTest accuracy:', test_acc)

# greater accuracy on training than test indicates overfitting

print("")

# fourth , make predictions
print("MAKING PREDICTIONS...")

# make predictions about some images via a softmax layer
# softmax layer converts the logits to probabilities - easier to interpret
probability_model = tf.keras.Sequential([model, \
                                        tf.keras.layers.Softmax()])

# prediction is an array of 10 numbers
# these numbers represent the model's confidence that the image corresponds to each of the 10 different articles of clothing
predictions = probability_model(test_images)
print(predictions[0])
print(np.sum(predictions[0]))
print(np.sum(predictions[0])*100)


# see which label has the highest confidence value (first image)
highest = np.argmax(predictions[0])
print(highest)

# examine the test label to see if the highest liklihood is correct
print(test_labels[0])

# plot image with labelled predicted and correct next to one another
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    
    if predicted_label == true_label:
        color='blue'
    else:
        color='red'

    # labels the image eg - 'ankle boot 98% (ankle boot)'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], \
                                            100*np.max(predictions_array), \
                                            class_names[true_label]), \
                                            color=color)

# plots a histogram of confidence probabilities for each label
# red if incorrect label
# blue if correct label
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])    
    
    thisplot = plt.bar(range(10), predictions_array, color="#777777")

    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# let's look at the 0th image
i = 0
plt.figure(figsize=(6,3))

plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)

plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)

# plt.show()
plt.close()

# # plot the first X test images, predicted labels, true labels, and prediction histograms
# # doesn't work - axis label issues, matplotlib nightmare
# num_rows = 5
# num_cols = 3
# num_images = num_rows * num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))

# for i in range(num_images):
#     plt.subplot(num_rows, 2*num_cols, 2*(i+1))
#     plot_image(i, predictions[i], test_labels, test_images)
    
#     plt.subplot(num_rows, 2*num_cols, 2*(i+2))
#     plot_value_array(i, predictions[i], test_labels)

# plt.tight_layout()
# plt.show()
# plt.close()


print("")
# fifth, use the trained model
print("USING TRAINED MODEL...")

# takeimage from test dataset
img = test_images[1]
print(img.shape)

# models are optimised to make predictions on a batch of examples at once
# even though predicting for single image, need to add it to a list
# adds the image to a batch where its the only member
img = (np.expand_dims(img, 0))

# now it is a (1, 28, 28)
print(img.shape)

# predict correct label for image
predictions_single = probability_model.predict(img)
print(predictions_single)

# let's look at the 0th image
i = 1
plt.figure(figsize=(6,3))

plt.subplot(1,2,1)
plot_image(i, predictions_single[0], test_labels, test_images)

plt.subplot(1,2,2)
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

# plt.show()
plt.close()

np.argmax(predictions_single[0])