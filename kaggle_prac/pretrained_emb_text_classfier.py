from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from keras.layers import Dense, Input, GlobalMaxPool1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint

MAX_SEQ_LEN = 200
MAX_NUM_WORDS = 20000
EMB_DIM = 100


train_df = pd.read_csv("D:/kaggle数据集/Toxic Comment Classification Challenge/train.csv")
test_df = pd.read_csv("D:/kaggle数据集/Toxic Comment Classification Challenge/test.csv")

train_comments = train_df["comment_text"].fillna("CVxTz").values
train_y = train_df.iloc[:, 2:].values
test_comments = test_df["comment_text"].fillna("CVxTz").values

# read pre_trained embedding matrix
embedding_index = {}
emb_file = open("D:/kaggle数据集/Toxic Comment Classification Challenge/glove.6B.100d.txt", 'r', encoding='utf-8')
for line in emb_file:
    values = line.split()
    word = values[0]
    wordvec = np.asarray(values[1:], dtype=np.float32)
    embedding_index[word] = wordvec
emb_file.close()

# preprocess comments
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(list(train_comments))
seq_train = tokenizer.texts_to_sequences(list(train_comments))
seq_test = tokenizer.texts_to_sequences(list(test_comments))
X_train = pad_sequences(seq_train, maxlen=MAX_SEQ_LEN)
X_test = pad_sequences(seq_test, maxlen=MAX_SEQ_LEN)

word_index = tokenizer.word_index

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index))
emb_matrix = np.zeros((num_words, EMB_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    emb_vector = embedding_index.get(word)
    if emb_vector is not None:
        emb_matrix[i] = emb_vector


def get_model():
    inp = Input(shape=(MAX_SEQ_LEN, ))
    x = Embedding(input_dim=num_words, output_dim=EMB_DIM,
                  weights=[emb_matrix], input_length=MAX_SEQ_LEN, trainable=False)(inp)
    x = Conv1D(filters=256, kernel_size=5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(64, 5, activation='relu')(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.2)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(0.2)(x)
    preds = Dense(units=train_y.shape[1], activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=preds)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = get_model()
batch_size = 128
epochs = 2
file_path = "pretrained_emb_model.hdf5"
model_checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
stopping = EarlyStopping(monitor='val_loss', mode='min', patience=20)
model.fit(X_train, train_y, batch_size=batch_size, epochs=epochs,
          validation_split=0.1, callbacks=[model_checkpoint, stopping])
model.load_weights(file_path)

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y_test = model.predict(X_test)
print(y_test[0])
submission_df = pd.read_csv("D:/kaggle数据集/Toxic Comment Classification Challenge/sample_submission.csv")
submission_df[list_classes] = y_test
submission_df.to_csv("submission.csv", index=False)
