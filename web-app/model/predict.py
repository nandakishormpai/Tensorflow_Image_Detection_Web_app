import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

reference = {0: "Rock", 1: "Paper", 2: "Scissors"}


def preprocess_img(image):
    image = tf.cast(image, tf.float32)
    image = image / 255.
    image = tf.image.resize(image, [256, 256])
    return image


def predict(image):
    model = load_model('model/rockpaper_model.h5')
    image = preprocess_img(image)
    result = np.argmax(model.predict(np.array([image])))
    return reference[result]
