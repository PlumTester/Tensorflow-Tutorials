import os
import PIL
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def main():

    load_dataset()
    
    return



def load_dataset():

    # download dataset
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

    # count total images
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)

    roses = list(data_dir.glob('roses/*'))
    # PIL.Image.open(str(roses[0])).show()
    # PIL.Image.close()

    # load using keras.preprocessing
    batch_size = 32
    img_height = 180
    img_width = 180

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir, \
                        validation_split=0.2, \
                        subset="training", \
                        seed=123, \
                        image_size=(img_height, img_width), \
                        batch_size=batch_size) 

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir, \
                        validation_split=0.2, \
                        subset="validation", \
                        seed=123, \
                        image_size=(img_height, img_width), \
                        batch_size=batch_size)

    class_names = train_ds.class_names

    # visualise
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")


    plt.show()
    plt.close()

    return


if __name__ == "__main__":
    main()



