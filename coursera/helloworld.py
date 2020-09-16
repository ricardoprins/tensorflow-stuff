import numpy as np
from tensorflow import keras


model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer="sgd", loss="mean_squared_error")
xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
ys = np.array([5.99999, 11.99999, 16.99999, 22.9999, 27.99999, 34.99999], dtype=float)
model.fit(xs, ys, epochs=500)
print(model.predict([7.0]))
