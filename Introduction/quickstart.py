# https://www.tensorflow.org/tutorials/quickstart/beginner
import tensorflow as tf
print("-----")
print("")

# load and prepare the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# convert the samples from integers to floats
x_train, x_test = x_train / 255.0, x_test / 255.0

# build the sequential model by stacking layers
# choose an optimiser and loss function for training
# model = tf.keras.models.Sequential([ \
#     tf.keras.layers.Flatten(input_shape=(28,28)), \
#     tf.keras.layers.Dense(128, activation='relu'), \
#     tf.keras.layers.Dropout(0.2), \
#     tf.keras.layers.Dense(10) \
# ])

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10))

# model description
# # 28x28 input kernel
# # 128 node fully connected layer
# # dropout layer with 0.2 dropout rate
# # 10 node fully connected layer

# note 10 node output where each node is a prediction classifier for digits 0-9

# for each input, model returns vector of logits (logg-odds) scores, one for each class
# puts the first (2?) data inputs through the model, prints predictions as a numpy array
predictions = model(x_train[:1]).numpy()
print(predictions)

# the softmax function converts these logits to probabilities for each class
print(tf.nn.softmax(predictions).numpy())

# takes vector of logits and True index
# returns a scalar loss for each example
# The scalar loss is equal to the negative log probability of the true class - is zero if model is sure of the correct class
# initial loss here should be close to 2.3 (tf.log(1/10)) - mnist has 10 classes 0-9
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# showing initial loss
print(loss_fn(y_train[:1], predictions). numpy())

# compile the model - chose adam optimiser, categorical crossentropy loss function
model.compile(optimizer='adam', \
                loss=loss_fn, \
                metrics=['accuracy'])

# fit the model - adjusts model parameters to minimise the loss
# chosen 5 epochs
model.fit(x_train, y_train, epochs=5)

# check the models performance, check on validation or test set
# outputs the processed examples, the loss, and accuracy
model.evaluate(x_test, y_test, verbose=2)

# if probability is preferred, wrap trained model and attach softmax to it
probability_model = tf.keras.Sequential([ \
    model, \
    tf.keras.layers.Softmax() \
])

print(probability_model(x_test[:5]))




