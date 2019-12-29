from __future__ import division, print_function, absolute_import

import os

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

train_df = pd.read_json("D:/kaggleDatas/C-CORE Iceberg Classifier Challenge/train.json")
test_df = pd.read_json("D:/kaggleDatas/C-CORE Iceberg Classifier Challenge/test.json")

train_brand1 = np.array([np.asarray(brand, np.float32).reshape(75, 75) for brand in train_df["band_1"]])
train_brand2 = np.array([np.asarray(brand, np.float32).reshape(75, 75) for brand in train_df["band_2"]])
x_train = np.concatenate([train_brand1[:, :, :, np.newaxis], train_brand2[:, :, :, np.newaxis],
                          ((train_brand1 + train_brand2) / 2)[:, :, :, np.newaxis]], axis=-1)
target_train = train_df['is_iceberg']


def create_model():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(75, 75, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def get_callbacks(filepath, patience=2):
    early_stopping = EarlyStopping('val_loss', patience=patience, mode='min')
    checkpoint = ModelCheckpoint(filepath, save_best_only=True)
    return [early_stopping, checkpoint]


file_path = ".model_weights.hdf5"
callbacks = get_callbacks(filepath=file_path, patience=5)

train_X, valid_X, train_y, valid_y = train_test_split(x_train, target_train, random_state=42, test_size=0.25)

image_data_generator = ImageDataGenerator(
    horizontal_flip=True, vertical_flip=True, width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1, rotation_range=40)

# image_data_generator.fit(train_X)

cnn_model = create_model()
cnn_model.fit_generator(image_data_generator.flow(train_X, train_y, batch_size=24, seed=42),
                        steps_per_epoch=len(train_X) / 24,
                        epochs=100, callbacks=callbacks, validation_data=(valid_X, valid_y))
# for epoch in range(100):
#     batches = 0
#     for x_batch, y_batch in image_data_generator.flow(train_X, train_y, batch_size=24):
#         loss = cnn_model.train_on_batch(x_batch, y_batch)
#         batches += 1
#         if batches >= len(x_train) // 24:
#             # we need to break the loop by hand because
#             # the generator loops indefinitely
#             break
# cnn_model.fit(train_X, train_y, batch_size=24, epochs=100, verbose=1, callbacks=callbacks,
#               validation_data=(valid_X, valid_y))

cnn_model.load_weights(filepath=file_path)
pred = cnn_model.evaluate(valid_X, valid_y, verbose=1)
print('Test loss:', pred[0])
print('Test acc:', pred[1])

test_brand1 = np.array([np.asarray(brand, np.float32).reshape(75, 75) for brand in test_df["band_1"]])
test_brand2 = np.array([np.asarray(brand, np.float32).reshape(75, 75) for brand in test_df["band_2"]])
x_test = np.concatenate([test_brand1[:, :, :, np.newaxis], test_brand2[:, :, :, np.newaxis],
                         ((test_brand1 + test_brand2) / 2)[:, :, :, np.newaxis]], axis=-1)
preds_test = cnn_model.predict_proba(x_test)

submission_df = pd.DataFrame()
submission_df['id'] = test_df['id']
submission_df['is_iceberg'] = preds_test.reshape((preds_test.shape[0]))
submission_df.to_csv('sub.csv', index=False)


