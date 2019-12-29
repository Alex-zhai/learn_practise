from __future__ import division, print_function, absolute_import
import os

import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input, Concatenate
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator, array_to_img
from keras.preprocessing.image import img_to_array, load_img
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_train_image(standardize=True):
    train_df = pd.read_csv("D:/kaggleDatas/leaf_classification/train.csv")
    train_id = train_df.pop("id")
    train_y = train_df.pop("species")
    train_y = LabelEncoder().fit(train_y).transform(train_y)
    train_x = StandardScaler().fit(train_df).transform(train_df) if standardize else train_df.values
    return train_id, train_x, train_y


def load_test_image(standardize=True):
    test_df = pd.read_csv("D:/kaggleDatas/leaf_classification/test.csv")
    test_id = test_df.pop("id")
    test_x = StandardScaler().fit(test_df).transform(test_df) if standardize else test_df.values
    return test_id, test_x


def resize_img(img, max_dim=96):
    if img.size[0] < img.size[1]:
        max_ax = 1
    else:
        max_ax = 0
    scale = max_dim / float(img.size[max_ax])
    return img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))


def load_image_data(image_id, max_dim=96, center=True):
    X = np.empty((len(image_id), max_dim, max_dim, 1))
    for i, id in enumerate(image_id):
        img = resize_img(load_img("D:/kaggleDatas/leaf_classification/images/{}.jpg".format(id), grayscale=True))
        img_array = img_to_array(img)
        length = img_array.shape[0]
        width = img_array.shape[1]
        if center:
            h1 = int((max_dim - length) / 2)
            h2 = h1 + length
            w1 = int((max_dim - width) / 2)
            w2 = w1 + width
        else:
            h1, w1 = 0, 0
            h2, w2 = (length, width)
        X[i, h1:h2, w1:w2, 0:1] = img_array
    return np.around(X / 255.0)


def load_train_data():
    train_id, train_x, train_y = load_train_image()
    print(train_x.shape, train_y.shape)
    train_img_x = load_image_data(train_id)
    shuffle_split = StratifiedShuffleSplit(n_splits=1, train_size=0.9, random_state=42)
    train_indices, valid_indices = next(shuffle_split.split(train_x, train_y))
    print(train_indices, valid_indices)
    valid_x, valid_x_img, valid_y = train_x[valid_indices], train_img_x[valid_indices], train_y[valid_indices]
    train_x, train_x_img, train_y = train_x[train_indices], train_img_x[train_indices], train_y[train_indices]
    return (train_x, train_x_img, train_y), (valid_x, valid_x_img, valid_y)


def load_test_data():
    test_id, test_x = load_test_image()
    test_img_x = load_image_data(test_id)
    return test_id, test_x, test_img_x


print('Loading the training data...')
(train_x, train_x_img, train_y), (valid_x, valid_x_img, valid_y) = load_train_data()
train_y_cat = to_categorical(train_y)
valid_y_cat = to_categorical(valid_y)
print('Training data loaded!')


class ImageDataGenerator2(ImageDataGenerator):
    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        return NumpyArrayIterator2(
            X, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)


class NumpyArrayIterator2(NumpyArrayIterator):
    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            # We changed index_array to self.index_array
            self.index_array = get_random_index_array(self.batch_size)
            current_batch_size = self.batch_size
            current_index = (self.batch_index * self.batch_size) % self.n
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.x.shape)[1:]))
        for i, j in enumerate(self.index_array):
            x = self.x[j]
            x = self.image_data_generator.random_transform(x.astype('float32'))
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1000),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        if self.y is None:
            return batch_x
        batch_y = self.y[self.index_array]
        return batch_x, batch_y


print('Creating Data Augmenter...')
imgen = ImageDataGenerator2(rotation_range=20, zoom_range=0.2, horizontal_flip=True, vertical_flip=True,
                            fill_mode='nearest')
imgen_train = imgen.flow(train_x_img, train_y_cat, seed=np.random.randint(1, 10000))

print('Finished making data augmenter...')


def get_random_index_array(batch_size):
    return np.random.randint(0, len(train_x_img), batch_size)


def create_model():
    input_image = Input(shape=(96, 96, 1), name='image')
    x = Conv2D(8, 5, strides=(5, 5), input_shape=(96, 96, 1), padding='same', activation='relu')(input_image)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Conv2D(32, 5, strides=(5, 5), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), (2, 2))(x)
    x = Flatten()(x)
    num_input = Input(shape=(192,), name='num')
    concat_layer = Concatenate()([x, num_input])
    x = Dense(100, activation='relu')(concat_layer)
    x = Dropout(0.5)(x)
    out = Dense(99, activation='softmax')(x)
    model = Model(inputs=[input_image, num_input], outputs=out)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def combined_generator(imgen, X):
    while True:
        for i in range(X.shape[0]):
            batch_img, batch_y = next(imgen)
            x = X[imgen.index_array]
            yield [batch_img, x], batch_y


best_model_file = "leafnet.h5"
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True)
stopping = EarlyStopping(monitor='val_loss', mode='min', patience=20)
callbacks = [best_model, stopping]

print('Training model...')
model = create_model()
model.fit_generator(combined_generator(imgen_train, train_x), steps_per_epoch=train_x.shape[0], epochs=89,
                    verbose=1,
                    callbacks=callbacks, validation_data=([valid_x_img, valid_x], valid_y_cat),
                    )
print("Loading the best model...")
model = load_model(best_model_file)
print("best model loaded!")

labels = sorted(pd.read_csv("D:/kaggleDatas/leaf_classification/train.csv").species.unique())
test_id, test_x, test_img_x = load_test_data()
preds = model.predict([test_img_x, test_x])
pred_df = pd.DataFrame(preds, index=test_id, columns=labels)
pred_df.to_csv("submission_leaf.csv")
print('Finished writing submission')
