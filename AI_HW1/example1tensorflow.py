import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#model = keras.Sequential(
#    [
#        keras.Input(shape=(2,)),
#        layers.Dense(2, activation="sigmoid", name="layer1"),
#        layers.Dense(2, activation="sigmoid", name="layer2"),
#    ]
#)

inputs = keras.Input(shape=(2,))
x = layers.Dense(2, activation="sigmoid")(inputs)
outputs = layers.Dense(2, activation="sigmoid", name="layer2")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

x_train = tf.constant([[1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0]])
y_train = tf.constant([[1.0, 0.0], [ 1.0,  0.0], [ 0.0, 1.0], [0.0,  1.0]])
#x_train = tf.expand_dims(x_train, axis=0)
#y_train = tf.expand_dims(y_train, axis=0)

model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=1.0),
    loss='MSE'
)

history = model.fit(
    x_train,
    y_train,
    epochs=1000
)

grid = np.linspace(-1.5, 1.5, 31)
x_val = np.transpose(np.meshgrid(grid, grid))
x_val = x_val.reshape((grid.size*grid.size,2))
y_val = model(x_val)
x_val = x_val.reshape((grid.size,grid.size,2))
y_val = tf.reshape(y_val, (grid.size,grid.size,2))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(x_val[:,:,0], x_val[:,:,1], y_val[:,:,0])
plt.show()

plt.plot(history.history['loss'])
plt.show()