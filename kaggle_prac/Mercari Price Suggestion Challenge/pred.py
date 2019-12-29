from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, \
    Embedding, Flatten, LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras import backend as K

MAX_NUM_WORDS = 20000
EMB_DIM = 100
MAX_NAME_SEQ = 10
MAX_DEC_SEQ = 100

train_df = pd.read_table("D:/kaggle数据集/Mercari Price Suggestion Challenge/train.tsv")
test_df = pd.read_table("D:/kaggle数据集/Mercari Price Suggestion Challenge/test.tsv")
# missing values
train_df.category_name.fillna(value='missing', inplace=True)
train_df.brand_name.fillna(value='missing', inplace=True)
train_df.item_description.fillna(value='missing', inplace=True)
test_df.category_name.fillna(value='missing', inplace=True)
test_df.brand_name.fillna(value='missing', inplace=True)
test_df.item_description.fillna(value='missing', inplace=True)

# handle category_name and brand_name
le = LabelEncoder()
le.fit(np.hstack([train_df.category_name, test_df.category_name]))
train_df.category_name = le.fit_transform(train_df.category_name)
test_df.category_name = le.fit_transform(test_df.category_name)

le.fit(np.hstack([train_df.brand_name, test_df.brand_name]))
train_df.brand_name = le.fit_transform(train_df.brand_name)
test_df.brand_name = le.fit_transform(test_df.brand_name)

# handle name and item_description
raw_text = np.hstack([train_df.name.str.lower(), train_df.item_description.str.lower()])
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(list(raw_text))
train_df['seq_name'] = tokenizer.texts_to_sequences(list(train_df.name.str.lower()))
test_df['seq_name'] = tokenizer.texts_to_sequences(list(test_df.name.str.lower()))
train_df['seq_item_description'] = tokenizer.texts_to_sequences((list(train_df.item_description.str.lower())))
test_df['seq_item_description'] = tokenizer.texts_to_sequences((list(test_df.item_description.str.lower())))

# read pre_trained embedding matrix
embedding_index = {}
emb_file = open("D:/kaggle数据集/Toxic Comment Classification Challenge/glove.6B.100d.txt", 'r', encoding='utf-8')
for line in emb_file:
    values = line.split()
    word = values[0]
    wordvec = np.asarray(values[1:], dtype=np.float32)
    embedding_index[word] = wordvec
emb_file.close()

# prepare embedding matrix
word_index = tokenizer.word_index
num_words = min(MAX_NUM_WORDS, len(word_index))
emb_matrix = np.zeros((num_words, EMB_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    emb_vector = embedding_index.get(word)
    if emb_vector is not None:
        emb_matrix[i] = emb_vector

MAX_CATE_LEN = np.max([train_df.category_name.max(), test_df.category_name.max()]) + 1
MAX_CATE_BRAND = np.max([train_df.brand_name.max(), test_df.brand_name.max()]) + 1
MAX_ITEM_COND = np.max([train_df.item_condition_id.max(), test_df.item_condition_id.max()]) + 1

train_df['target'] = np.log(train_df.price + 1)
scaler = MinMaxScaler(feature_range=(-1, 1))
train_df['target'] = scaler.fit_transform(train_df.target.reshape(-1, 1))


def get_keras_data(dataset):
    data = {
        'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ),
        'item_des': pad_sequences(dataset.seq_item_description, maxlen=MAX_DEC_SEQ),
        'brand_name': np.array(dataset.brand_name),
        'category_name': np.array(dataset.category_name),
        'item_condition': np.array(dataset.item_condition_id),
        'shipping': np.array(dataset[['shipping']])
    }
    return data


X_train = get_keras_data(train_df)
X_test = get_keras_data(test_df)


def rmsle_cust(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))


def get_callbacks(filepath, patience=2):
    stopping = EarlyStopping(monitor='val_loss', patience=patience, mode='min')
    checkpoint = ModelCheckpoint(filepath=filepath, save_best_only=True)
    return [stopping, checkpoint]


def get_model():
    name = Input(shape=[X_train['name'].shape[1]], name='name')
    item_des = Input(shape=[X_train['item_des'].shape[1]], name='item_des')
    brand_name = Input(shape=[1], name='brand_name')
    category_name = Input(shape=[1], name='category_name')
    item_condition = Input(shape=[1], name='item_condition')
    shipping = Input(shape=[X_train['shipping'].shape[1]], name='shipping')

    # embedding layers
    name_emb = Embedding(input_dim=num_words, output_dim=EMB_DIM, weights=[emb_matrix],
                         input_length=MAX_NAME_SEQ, trainable=False)(name)
    item_des_emb = Embedding(input_dim=num_words, output_dim=EMB_DIM, weights=[emb_matrix],
                             input_length=MAX_DEC_SEQ, trainable=False)(item_des)
    brand_name_emb = Embedding(input_dim=MAX_CATE_BRAND, output_dim=10)(brand_name)
    category_name_emb = Embedding(input_dim=MAX_CATE_LEN, output_dim=10)(category_name)
    item_condition_emb = Embedding(input_dim=MAX_ITEM_COND, output_dim=5)(item_condition)

    rnn_layer1 = LSTM(16)(item_des_emb)
    rnn_layer2 = LSTM(8)(name_emb)

    # main layer
    main_layer = concatenate([
        Flatten()(brand_name_emb),
        Flatten()(category_name_emb),
        Flatten()(item_condition_emb),
        rnn_layer1,
        rnn_layer2,
        shipping
    ])

    main_layer = Dense(128)(main_layer)
    main_layer = Dropout(0.1)(main_layer)
    main_layer = Dense(64)(main_layer)
    main_layer = Dropout(0.1)(main_layer)

    output = Dense(1, activation='linear')(main_layer)
    model = Model(inputs=[name, item_des, brand_name, category_name, item_condition, shipping], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae', rmsle_cust])
    return model


model = get_model()
model.summary()

batch_size = 20000
epochs = 5
file_path = "pretrained_emb_model.hdf5"
patience = 10
model.fit(X_train, train_df.target, batch_size=batch_size, epochs=epochs, verbose=1,
          callbacks=get_callbacks(filepath=file_path, patience=patience), validation_split=0.1)

model.load_weights(file_path)
y_preds = model.predict(X_test)
y_preds = scaler.inverse_transform(y_preds)
y_preds = np.exp(y_preds) - 1

submission_df = pd.read_csv("D:/kaggle数据集/Toxic Comment Classification Challenge/sample_submission.csv")
submission_df['price'] = y_preds
submission_df.to_csv("submission.csv", index=False)
