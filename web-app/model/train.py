import os
import numpy as np
from keras.models import Model
from keras.layers import BatchNormalization, Dropout
from keras.layers import GlobalAveragePooling2D, Dense
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.vgg16 import VGG16
from lrfinder import LRFinder
K = tf.keras.backend

DATASET_NAME = 'rock_paper_scissors'

(dataset_train_raw, dataset_test_raw), dataset_info = tfds.load(
    name=DATASET_NAME,
    data_dir='tmp',
    with_info=True,
    as_supervised=True,
    split=[tfds.Split.TRAIN, tfds.Split.TEST],
)


def preprocess_img(image, label):
    # Make image color values to be float.
    image = tf.cast(image, tf.float32)
    # Make image color values to be in [0..1] range.
    image = image / 255.
    # Make sure that image has a right size
    image = tf.image.resize(image, [256, 256])
    return image, label


dataset_train = dataset_train_raw.map(preprocess_img)
dataset_test = dataset_test_raw.map(preprocess_img)

dataset_train = dataset_train.map(
    lambda image, label: (
        tf.image.convert_image_dtype(image, tf.float32), label)
).cache().map(
    lambda image, label: (tf.image.random_flip_left_right(image), label)
).map(
    lambda image, label: (tf.image.random_contrast(
        image, lower=0.0, upper=1.0), label)
).shuffle(
    100
).batch(
    64
).repeat()

dataset_test = dataset_test.batch(32)


vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=(256, 256, 3))
vgg16.trainable = False


x = vgg16.output

x = GlobalAveragePooling2D()(x)

x = BatchNormalization()(x)

x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)


x = Dense(3, activation='softmax')(x)

model = Model(vgg16.input, x)

model.compile(optimizer='Adam',
              loss="sparse_categorical_crossentropy",
              metrics=["sparse_categorical_accuracy"])

earlystop_callback = EarlyStopping(
    monitor='val_loss', min_delta=0.0001, patience=2)

history = model.fit(dataset_train, epochs=20, steps_per_epoch=32,
                    validation_data=dataset_test, callbacks=[earlystop_callback])

model.save('rockpaper_model.h5')
