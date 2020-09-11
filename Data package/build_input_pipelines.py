# https://www.tensorflow.org/guide/data

# https://www.tensorflow.org/api_docs/python/tf/data/Dataset


# build tensorflow input pipelines

# two ways to create a dataset
#   data source - constructs Dataset from data stored in memory or in one or more files
#   data transformation constructs a dataset from one or more tf.data.Datasets

import tensorflow as tf

import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import scipy.ndimage as ndimage


def main():
    
    # basic_mechanics()

    # dataset_structure()

    # basic_transformation()

    # reading_input_data_in_memory()

    # image_data_generator()

    # load_pandas_using_dicts()

    # consuming_sets_of_files()

    # simple_batching()

    # batching_with_padding()

    # repeat_use_process_multiple_epochs_of_same_data()

    # shuffle_input_data()

    # # basic parse and show images using map
    # decode_resize_image()

    # python_logic_on_images()

    # time_series_windowing_using_batch()

    # time_series_windowing()

    # good make_window_dataset function
    time_series_windowing_complete()

    return 



def time_series_windowing_complete():

    range_ds = tf.data.Dataset.range(100000)

    def make_window_dataset(ds, window_size=5, shift=1, stride=1):
        windows = ds.window(window_size, shift=shift, stride=stride)

        def sub_to_batch(sub):
            return sub.batch(window_size, drop_remainder=True)

        windows = windows.flat_map(sub_to_batch)
        return windows

    ds = make_window_dataset(range_ds, window_size=10, shift=5, stride=3)

    for example in ds.take(10):
        print(example.numpy())

    print('-')

    # extract labels
    # may shift featuers and labels by one step relative to each other
    def dense_1_step(batch):
        # shift features and labels one step relative to each other
        return batch[:-1], batch[1:]

    dense_labels_ds = ds.map(dense_1_step)

    for inputs, labels in dense_labels_ds.take(3):
        print(inputs.numpy(), " => ", labels.numpy())

    print('-')


    return



def time_series_windowing():

    range_ds = tf.data.Dataset.range(100000)

    window_size = 10

    windows = range_ds.window(window_size, shift=1)

    
    # basic process
    #   window
    #   flat map
    #   batch with same as window size - as the flat map fn


    for sub_ds in windows.take(5):
        print(sub_ds)

    print('-')

    # dataset.flat_map will take dataset of datasets and flatten it into a single dataset
    for x in windows.flat_map(lambda x: x).take(30):
        print(x.numpy(), end=' ')

    print('-')

    # almost always want to batch dataset first
    def sub_to_batch(sub):
        return sub.batch(window_size, drop_remainder=True)

    for example in windows.flat_map(sub_to_batch).take(5):
        print(example.numpy())

    print('-')



    return



def time_series_windowing_using_batch():

    range_ds = tf.data.Dataset.range(100000)

    # typically want contiguous time slice

    simple_approach_batches = range_ds.batch(10, drop_remainder=True)
    for batch in simple_approach_batches.take(5):
        print(batch.numpy())

    # may shift featuers and labels by one step relative to each other
    def dense_1_step(batch):
        # shift features and labels one step relative to each other
        return batch[:-1], batch[1:]

    # this shifts window to be predicted (dense_1_step_df) by 1 so that it includes time t+1
    dense_1_step_ds = simple_approach_batches.map(dense_1_step)

    for features, label in dense_1_step_ds.take(3):
        print(features.numpy(), " => ", label.numpy())

    print('-')

    # to predict whole window instead of fixed offset, you can split batches into two parts
    batches = range_ds.batch(15, drop_remainder=True)

    def label_next_5_steps(batch):
        # returns first 5 steps and the remained from first five steps
        return (batch[:-5], batch[-5:])

    # has 10 predict the next 5
    dense_5_steps = batches.map(label_next_5_steps)

    for features, label in dense_1_step_ds.take(3):
        print(features.numpy(), " => ", label.numpy())

    print('-')
    
    
    
    return



def python_logic_on_images():

    def random_rotate_image(image):
        image = ndimage.rotate(image, np.random.uniform(-30, 30), reshape=False)
        return image
        
    # when operating on image
    # MUST describe return shapes and types when you apply the function
    def tf_random_rotate_image(image, label):
        im_shape = image.shape
        [image,] = tf.py_function(random_rotate_image, [image], [tf.float32])
        image.set_shape(im_shape)
        return image, label


    def show(image, label):
        plt.figure()
        plt.imshow(image)
        plt.title(label.numpy().decode('utf-8'))
        plt.axis('off')
        plt.show()
        plt.close()
        return


    # write function that manipulates the dataset elements
    def parse_image(filename):
        parts = tf.strings.split(filename, os.sep)
        label = parts[-2]

        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)

        image = tf.image.resize(image, [128, 128])
        
        return image, label


    #rebuild flower filenames dataset
    flowers_root = tf.keras.utils.get_file('flower_photos', \
                                            'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', \
                                            untar=True)
            
    flowers_root = pathlib.Path(flowers_root)
    list_ds = tf.data.Dataset.list_files(str(flowers_root/'*/*'))

    # map it over the dataset
    images_ds = list_ds.map(parse_image)

    image, label = next(iter(images_ds))
    image = random_rotate_image(image)
    show(image, label)

    # using better rotate function with shape definition
    rot_ds = images_ds.map(tf_random_rotate_image)
    for image, label in rot_ds.take(2):
        show(image, label)


    return



# how to use map
def decode_resize_image():

    # dataset.map(f) transformation applies function f to each element of input dataset

    
    #rebuild flower filenames dataset
    flowers_root = tf.keras.utils.get_file('flower_photos', \
                                            'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', \
                                            untar=True)
            
    flowers_root = pathlib.Path(flowers_root)
    list_ds = tf.data.Dataset.list_files(str(flowers_root/'*/*'))


    # necessary to convert images of different sizes to common size (batched into fixed size)

    # write function that manipulates the dataset elements
    def parse_image(filename):
        parts = tf.strings.split(filename, os.sep)
        label = parts[-2]

        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)

        image = tf.image.resize(image, [128, 128])
        
        return image, label

    # test parse function
    file_path = next(iter(list_ds))
    image, label = parse_image(file_path)

    def show(image, label):
        plt.figure()
        plt.imshow(image)
        plt.title(label.numpy().decode('utf-8'))
        plt.axis('off')
        plt.show()
        plt.close()
        return

    show(image, label)

    # map it over the dataset
    images_ds = list_ds.map(parse_image)

    for image, label in images_ds.take(2):
        show(image, label)

    return



# maintains fixed size buffer and chooses next element uniformly at random from that buffer
def shuffle_input_data():

    titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
    titanic_lines = tf.data.TextLineDataset(titanic_file)

    # add index to dataset so you can see effect
    lines = tf.data.TextLineDataset(titanic_file)
    counter = tf.data.experimental.Counter()

    dataset = tf.data.Dataset.zip((counter, lines))
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(20)

    # note buffer size 100, batch size 20
    # 20*200 (no elements with index of 120)
    print(dataset)
    n, line_batch = next(iter(dataset))
    print(n.numpy())

    print('-')

    # order relative to repeat matters
    # shuffle doesn't signal end of epoch until shuffle buffer is empty
    # shuffle placed before a repeat will show every element of one epoch before moving to the next
    dataset = tf.data.Dataset.zip((counter, lines))

    shuffled = dataset.shuffle(buffer_size=100).batch(10).repeat(2)

    # OVERALL - THIS IS WHAT YOU WANT
    # SHUFFLE.BATCH.REPEAT
    # SHUFFLES, then batches clear epoch boundaires with repeat
    print("Here are the item ID's near the epoch boundary:\n")
    for n, line_batch in shuffled.skip(60).take(5):
        print(n.numpy())

    print('-')

    shuffle_repeat = [n.numpy().mean() for n, line_batch in shuffled]
    plt.plot(shuffle_repeat, label="shuffle().repeat()")
    plt.ylabel("Mean item ID")
    plt.legend()
    plt.show()
    plt.close()

    # repeat before shuffle mixes epoch boundaries together
    dataset = tf.data.Dataset.zip((counter, lines))
    shuffled = dataset.repeat(2).shuffle(buffer_size=100).batch(10)

    print("Here are the item ID's near the epoch boundary:\n")
    for n, line_batch in shuffled.skip(55).take(15):
        print(n.numpy())

    repeat_shuffle = [n.numpy().mean() for n, line_batch in shuffled]

    plt.plot(shuffle_repeat, label="shuffle().repeat()")
    plt.plot(repeat_shuffle, label="repeat().shuffle()")
    plt.ylabel("Mean item ID")
    plt.legend()
    plt.show()
    plt.close()

    shuffled = dataset.batch(10).repeat(2).shuffle(buffer_size=100)

    print("Here are the item ID's near the epoch boundary:\n")
    for n, line_batch in shuffled.skip(55).take(15):
        print(n.numpy())

    batch_shuffle = [n.numpy().mean() for n, line_batch in shuffled]

    plt.plot(shuffle_repeat, label="shuffle().batch().repeat()")
    plt.plot(batch_shuffle, label="batch().shuffle().repeat()")
    plt.plot(repeat_shuffle, label="repeat().shuffle().batch()")
    plt.ylabel("Mean item ID")
    plt.legend()
    plt.show()
    plt.close()
    
    # batch before repeat, then shuffle
    # should have clear boundaries and shuffle fine
    shuffled = dataset.batch(10).repeat(2).shuffle(buffer_size=100)

    print("Here are the item ID's near the epoch boundary:\n")
    for n, line_batch in shuffled.skip(55).take(15):
        print(n.numpy())

    batch_repeat_shuffle = [n.numpy().mean() for n, line_batch in shuffled]

    plt.plot(shuffle_repeat, label="shuffle().batch().repeat()")
    # plt.plot(batch_shuffle, label="batch().shuffle().repeat()")
    # plt.plot(repeat_shuffle, label="repeat().shuffle().batch()")
    plt.plot(batch_repeat_shuffle, label="batch().repeat().shuffle()")
    plt.ylabel("Mean item ID")
    plt.legend()
    plt.show()
    plt.close()

    return


# repeat tutorial
def repeat_use_process_multiple_epochs_of_same_data():

    titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
    titanic_lines = tf.data.TextLineDataset(titanic_file)

    def plot_batch_sizes(ds):
        batch_sizes = [batch.shape[0] for batch in ds]
        plt.bar(range(len(batch_sizes)), batch_sizes)
        plt.xlabel('Batch number')
        plt.ylabel('Batch size')

    # repeat concatenates arguments without signaling the end of one epoch and the beginning of the next epoch
    # because of this - batch is applied after repeat
    #   this yields batches that straddle epoch boundaries

    # applying repeat transformation with no arguments will repeat input indefinitely

    # no repeat - 4 batches
    titanic_batches = titanic_lines.batch(128)
    plot_batch_sizes(titanic_batches)
    plt.show()
    plt.close()
    
    # repeat before batch - batch straddles epoch boundaries
    # 13.6 batches
    titanic_batches = titanic_lines.repeat(3).batch(128)
    plot_batch_sizes(titanic_batches)
    plt.show()
    plt.close()

    # batch before repeat - clear epoch boundaries on batch
    # 14 batches, but data didn't overflow for uncomplete remainder
    titanic_batches = titanic_lines.batch(128).repeat(3)
    plot_batch_sizes(titanic_batches)
    plt.show()
    plt.close()

    # basic repeat feeds data through for each epoch
    # repeat should be number of epochs

    # custom computation example at the end of each epoch
    # restart dataset iteration on each epoch
    epochs = 3
    dataset = titanic_lines.batch(128)

    for epoch in range(epochs):
        for batch in dataset:
            print(batch.shape)

        print('End of epoch: ', epoch)

    return



# many models work with input data that can have varying size
# padding transformaiton enables batch tensors of different shape by specifying one or more dimensions in which they may be padded
def batching_with_padding():

    dataset = tf.data.Dataset.range(100)
    dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
    dataset = dataset.padded_batch(4, padded_shapes=(None,))

    for batch in dataset.take(2):
        print(batch.numpy())
        print()

    # set different padding for each dimension of each component
    #   None means variable length
    #   may be a constant

    return



# stacks n consecutive elements of a dataset into a single element
# all elements must have a tensor of the exact same shape
def simple_batching():

    inc_dataset = tf.data.Dataset.range(100)
    dec_dataset = tf.data.Dataset.range(0, -100, -1)

    # print(inc_dataset.element_spec)
    # for e in inc_dataset:
    #     print(e)
    # print('-')

    # print(dec_dataset.element_spec)
    # for e in dec_dataset:
    #     print(e)
    # print('-')

    dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
    for a, b in dataset:
        print('{}, {}'.format(a, b))

    print('-')

    batched_dataset = dataset.batch(4) 
    
    print(batched_dataset.element_spec)
    for a, b in batched_dataset.take(4):
        print('{}, {}'.format(a, b))

    print('-')

    # default settings of batch result is an unknown batch size because last batch may not be full
    # use drop_remainder arugment to ignore that last batch, get full shape propagation
    batched_dataset = dataset.batch(7, drop_remainder=True)
    print(batched_dataset.element_spec)
    for a, b in batched_dataset.take(4):
        print('{}, {}'.format(a, b))

    print('-')

    return



def consuming_sets_of_files():

    flowers_root = tf.keras.utils.get_file('flower_photos', \
                                            'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', \
                                            untar=True)
            
    flowers_root = pathlib.Path(flowers_root)

    # root directory contains directory for each class
    for item in flowers_root.glob("*"):
        print(item.name)

    print('-')

    # files in each class directory are examples
    list_ds = tf.data.Dataset.list_files(str(flowers_root/'*/*'))

    for f in list_ds.take(5):
        print(f.numpy())

    print("-")

    # read data using tf.io.read_file function
    # extract label from path, returning (image, label) pairs
    def process_path(file_path):
        label = tf.strings.split(file_path, os.sep)[-2]
        return tf.io.read_file(file_path), label

    labeled_ds = list_ds.map(process_path)

    for image_raw, label_text in labeled_ds:
        print(repr(image_raw.numpy()[:100]))
        print()
        print(label_text.numpy())
        print('-')

    print('-')

    return


def load_pandas_using_dicts():
    # easiest way to preserve column structure of pd.DF is to convert df to dict and slice the dict
    csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/applied-dl/heart.csv')
    df = pd.read_csv(csv_file)

    target = df.pop('target')

    dict_slices = tf.data.Dataset.from_tensor_slices((df.to_dict('list'), target.values)).batch(16)
    
    # print each dict slice of 16 data points
    for dict_slice in dict_slices.take(1):
        print (dict_slice)



def load_pandas_dataframe():
    # https://www.tensorflow.org/tutorials/load_data/pandas_dataframe

    csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/applied-dl/heart.csv')
    df = pd.read_csv(csv_file)

    target = df.pop('target')

    dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

    for feat, targ in dataset.take(5):
        print ('Features: {}, Target: {}'.format(feat, targ))

    # shuffle train (batch=1)
    train_dataset = dataset.shuffle(len(df)).batch(1)

    # define model...
    model = None

    # train
    model.fit(train_dataset, epochs=15)


# uses batching to read images from directory
def image_data_generator():

    flowers = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)

    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20)

    images, labels = next(img_gen.flow_from_directory(flowers))

    print(images.dtype, images.shape)
    print(labels.dtype, labels.shape)

    ds = tf.data.Dataset.from_generator(img_gen.flow_from_directory, args=[flowers], \
                                            output_types=(tf.float32, tf.float32), \
                                            output_shapes=([32,256,256,3], [32,5]))

    print(ds)

    return


def reading_input_data_in_memory():

    # simple all fit in memory
    train, test = tf.keras.datasets.fashion_mnist.load_data()

    images, labels = train
    images = images/255

    print(images)
    print(labels)

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    print(dataset)


def basic_transformation():

    dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10], minval=1, maxval=10, dtype=tf.int32))

    dataset2 = tf.data.Dataset.from_tensor_slices((tf.random.uniform([4]), tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))

    dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
    print(dataset3)
    print(dataset3.element_spec)
    for e in dataset3:
        print(e)
    print('-')
    print('-')

    # for 4 x 10 x 100
    # a is n=4
    # b is n=10
    # c is n=100
    for a, (b, c) in dataset3:
        print('shapes: {a.shape}, {b.shape}, {c.shape}'.format(a=a, b=b, c=c))
        print(a)
        print(b)
        print(c)
        print('-')

    return


def dataset_structure():

    # dataset contains elements that have the same (nested) structure and individual components of structure can be any type of tf.TensorArray
    #   tf.Tensor, tf.sparse.SpareTensor, etc

    # inspect type of element component using Dataset.element_spec
    dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))
    print(dataset1.element_spec)
    for elem in dataset1:
        print(elem)
    print('-')

    dataset2 = tf.data.Dataset.from_tensor_slices((tf.random.uniform([4]), tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))
    print(dataset2.element_spec)
    for elem in dataset2:
        print(elem)
    print('-')

    dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
    print(dataset3.element_spec)
    for elem in dataset3:
        print(elem)
    print('-')

    return



def basic_mechanics():
        
    # start with a data source - eg construct Dataset from data in memory
    # can use .from_tensors() or .from_tensor_slices()
    # can use TFRecordDataset() if stored in recommended file    
    dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])

    # Dataset object can transform into new dataset using tf.data.Dataset
    #   eg apply per-element transformations eg Dataset.map()
    #   eg apply multi-element transformations such as Dataset.batch()

    # object is Python iterable - makes it possible to consume elements using a for loop
    for elem in dataset:
        print(elem.numpy())

    print("-")

    # can explicitly create Python iterator using iter and consuming elements using next
    it = iter(dataset)
    print(next(it).numpy())

    print("-")

    # can consume elements using reduce transformation
    # reduces all elements to produce single result
    # this example shows how to reduce transformation to compute sum
    print(dataset.reduce(0, lambda state, value: state+value).numpy())
    print("-")

    return


    







if __name__ == "__main__":
    main()
