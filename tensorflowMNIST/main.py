from keras import layers
from keras import optimizers
from keras.callbacks import TensorBoard
from matplotlib import pyplot

from tensorflow import keras
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
x_test = x_test.astype('float32')
y_test = y_test.astype('float32')

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

input_shape = x_train.shape[1:]
n_classes = len(np.unique(y_train))

y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)

model = keras.Sequential()
model.add(layers.BatchNormalization(input_shape=input_shape))
model.add(layers.Conv2D(filters=3, kernel_size=5, strides=1))
model.add(layers.MaxPool2D(pool_size=4, strides=1))
model.add(layers.Activation(activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=9, kernel_size=3, strides=1))
model.add(layers.MaxPool2D(pool_size=4, strides=1))
model.add(layers.Activation(activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(500))
model.add(layers.Activation(activation="relu"))
model.add(layers.Dense(n_classes))
model.add(layers.Activation(activation="softmax"))
model.build()

epochs = 15
num_batch = 32
optimizer = optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])

log_folder = "logtf/fit"
callbacks = [TensorBoard(log_dir=log_folder,
                         histogram_freq=1,
                         write_graph=True,
                         write_images=True,
                         update_freq='epoch',
                         profile_batch=2,
                         embeddings_freq=1)]

history = model.fit(x_train, y_train, epochs=epochs, batch_size=num_batch, validation_split=0.3, callbacks=callbacks,
                    verbose=1)

"""
pyplot.title("Learning losses")
pyplot.xlabel("epochs")
pyplot.ylabel("loss")
pyplot.plot(history.history['val_loss'], 'r', label="Validation loss")
pyplot.plot(history.history['loss'], 'b', label="Training loss")
pyplot.legend()
pyplot.show()
"""

model.evaluate(x_test, y_test, batch_size=len(x_test), verbose=1)


version = 1
file_path = f"./img_classifier/{version}/"
model.save(filepath=file_path, save_format='tf')
