from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

np.random.seed(42)

batch_size = 128
image_height = 28
image_width = 28
nb_classes = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(K.image_dim_ordering())
if K.image_dim_ordering() == 'th':
    x_train = x_train.reshape(x_train.shape[0], 1, image_height, image_width)
    x_test = x_test.reshape(x_test.shape[0], 1, image_height, image_width)
    input_shape = (1, image_height, image_width)
else:
    x_train = x_train.reshape(x_train.shape[0], image_height, image_width, 1)
    x_test = x_test.reshape(x_test.shape[0], image_height, image_width, 1)
    input_shape = (image_height, image_width, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = np_utils.to_categorical(y_train, num_classes=nb_classes)
y_test = np_utils.to_categorical(y_test, num_classes=nb_classes)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='valid', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(filters=32, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=20, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])