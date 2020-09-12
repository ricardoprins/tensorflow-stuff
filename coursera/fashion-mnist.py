import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0


def showImage(int):
    np.set_printoptions(linewidth=200)
    plt.imshow(train_images[int])
    print(train_labels[int])
    print(train_images[int])


class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get("loss")) < 0.4:
            self.model.stop_training = True


def fashionMNISTmodel():
    callbacks = myCallback()
    model = keras.models.Sequential(
        [
            keras.layers.Flatten(),
            keras.layers.Dense(1024, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(train_images, train_labels, epochs=1, callbacks=[callbacks])
    model.evaluate(test_images, test_labels)
    classifications = model.predict(test_images)
    print(classifications[0])
    print(test_labels[0])


fashionMNISTmodel()