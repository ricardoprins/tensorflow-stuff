# dataset link: https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

train_horse_dir = os.path.join("datasets/horses-or-humans/horses")
train_human_dir = os.path.join("datasets/horses-or-humans/humans")
valid_horse_dir = os.path.join("datasets/valid-horses-or-humans/horses")
valid_human_dir = os.path.join("datasets/valid-horses-or-humans/humans")

train_horse_names = os.listdir(train_horse_dir)
train_human_names = os.listdir(train_human_dir)
valid_horse_names = os.listdir(valid_horse_dir)
valid_human_names = os.listdir(valid_human_dir)
nrows = 4
ncols = 4
pic_index = 0

fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_horse_pic = [
    os.path.join(train_horse_dir, fname)
    for fname in train_horse_names[pic_index - 8 : pic_index]
]
next_human_pic = [
    os.path.join(train_human_dir, fname)
    for fname in train_human_names[pic_index - 8 : pic_index]
]

for i, img_path in enumerate(next_horse_pic + next_human_pic):
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis("Off")
    img = mpimg.imread(img_path)
    plt.imshow(img)

# plt.show()

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            16, (3, 3), activation="relu", input_shape=(300, 300, 3)
        ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.summary()

model.compile(
    loss="binary_crossentropy", optimizer=RMSprop(lr=0.001), metrics=["accuracy"]
)

train_datagen = ImageDataGenerator(rescale=1 / 255)
valid_datagen = ImageDataGenerator(rescale=1 / 255)
train_generator = train_datagen.flow_from_directory(
    "datasets/horses-or-humans/",
    target_size=(300, 300),
    batch_size=128,
    class_mode="binary",
)
valid_generator = valid_datagen.flow_from_directory(
    "datasets/valid-horses-or-humans/",
    target_size=(300, 300),
    batch_size=32,
    class_mode="binary",
)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    verbose=1,
    validation_data=valid_generator,
    validation_steps=8,
)
