from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import keras
import numpy as np
import pandas as pd
from PIL import Image
from keras.layers import Dense, Dropout, Flatten, ZeroPadding2D, Convolution2D, MaxPooling2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

df_train = pd.read_csv("D:/kaggleDatas/dog image classfication/labels.csv")
df_test = pd.read_csv("D:/kaggleDatas/dog image classfication/sample_submission.csv")

print(df_train.head(10))

target_series = pd.Series(df_train['breed'])
one_hot = pd.get_dummies(target_series, sparse=True)
one_hot_labels = np.asarray(one_hot)
# print(one_hot_labels)

im_size = 90
x_train = []
y_train = []
x_test = []

i = 0
for imgId, bleed in tqdm(df_train.values):
    img = Image.open('D:/kaggleDatas/dog image classfication/train/{}.jpg'.format(imgId))
    label = one_hot_labels[i]
    img.resize((im_size, im_size))
    pixes = np.zeros((im_size, im_size, 3), np.float32)
    pix = img.load()
    for x in range(im_size):
        for y in range(im_size):
            pixes[x][y] = pix[x, y]
    x_train.append(pixes)
    y_train.append(label)
    i += 1

for imgId in tqdm(df_test['id'].values):
    img = Image.open('D:/kaggleDatas/dog image classfication/test/{}.jpg'.format(imgId))
    img.resize((im_size, im_size))
    pixes = np.zeros((im_size, im_size, 3), np.float32)
    pix = img.load()
    for x in range(im_size):
        for y in range(im_size):
            pixes[x][y] = pix[x, y]
    x_test.append(pixes)

x_train_raw = np.array(x_train, np.float32) / 255.
y_train_raw = np.array(y_train, np.uint8)
x_test = np.array(x_test, np.float32) / 255.

print(x_train_raw.shape, y_train_raw.shape, x_test.shape)

num_class = y_train_raw.shape[1]

X_train, X_valid, Y_train, Y_valid = train_test_split(x_train_raw, y_train_raw, test_size=0.3, random_state=42)


def base_vgg19():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(im_size, im_size, 3)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_class, activation='softmax'))
    return model


base_model = base_vgg19()

base_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
base_model.fit(X_train, Y_train, epochs=10, validation_data=(X_valid, Y_valid), callbacks=callbacks_list, verbose=1)

preds = base_model.predict(x_test, verbose=1)

submission_df = pd.DataFrame(preds)
col_names = one_hot.columns.values
submission_df.columns = col_names
submission_df.insert(0, 'id', df_test['id'])
print(submission_df.head(5))

submission_df.to_csv("submission.csv", index=False)
