from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import re
from keras.layers import Dense, Input, GlobalMaxPool1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, SpatialDropout1D, Concatenate
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# pre_process text data
repl = {
    "yay!": " good ",
    "yay": " good ",
    "yaay": " good ",
    "yaaay": " good ",
    "yaaaay": " good ",
    "yaaaaay": " good ",
    "&lt;3": " heart ",
    ":d": " smile ",
    ":p": " smile ",
    ":dd": " smile ",
    "8)": " smile ",
    ":-)": " smile ",
    ":)": " smile ",
    ";)": " smile ",
    "(-:": " smile ",
    "(:": " smile ",
    ":/": " worry ",
    ":&gt;": " angry ",
    ":')": " sad ",
    ":-(": " sad ",
    ":(": " sad ",
    ":s": " sad ",
    ":-s": " sad ",
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not",
    r"\bi'm\b": "i am",
    "r": "are",
    "u": "you",
    "haha": "ha",
    "hahaha": "ha",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "can't": "can not",
    "cannot": "can not",
    "i'm": "i am",
    "m": "am",
    "i'll": "i will",
    "its": "it is",
    "it's": "it is",
    "'s": " is",
    "that's": "that is",
    "weren't": "were not",
}

keys = [key for key in repl.keys()]

train_comments = train_df["comment_text"].tolist()
test_comments = test_df["comment_text"].tolist()


def process_text(text_datas):
    new_text_datas = []
    for text in text_datas:
        split_text = str(text).split()
        new_text = ""
        for w in split_text:
            w = str(w).lower()
            if w[:4] == 'http' or w[:3] == 'www':
                continue
            if w in keys:
                w = repl[w]
            new_text += w + " "
        new_text_datas.append(new_text)
    return new_text_datas


new_train_data = process_text(train_comments)
new_test_data = process_text(test_comments)

train_df["new_comment_text"] = new_train_data
test_df["new_comment_text"] = new_test_data

print("crap removed")
trate = train_df["new_comment_text"].tolist()
tete = test_df["new_comment_text"].tolist()

for i, c in enumerate(trate):
    trate[i] = re.sub('[^a-zA-Z ?!]+', '', str(trate[i]).lower())

for i, c in enumerate(tete):
    tete[i] = re.sub('[^a-zA-Z ?!]+', '', tete[i])

train_df["comment_text"] = trate
test_df["comment_text"] = tete
print('only alphabets')

X_train = train_df["comment_text"].fillna("fillna").values
y_train = train_df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_test = test_df["comment_text"].fillna("fillna").values

# read pre_trained embedding matrix
embedding_index = {}
emb_file = open("glove.6B.300d.txt")
for line in emb_file:
    values = line.split()
    word = values[0]
    wordvec = np.asarray(values[1:], dtype=np.float32)
    embedding_index[word] = wordvec
emb_file.close()

# preprocess comments
MAX_SEQ_LEN = 100
MAX_NUM_WORDS = 30000
EMB_DIM = 300

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
seq_train = tokenizer.texts_to_sequences(X_train)
seq_test = tokenizer.texts_to_sequences(X_test)
X_train = pad_sequences(seq_train, maxlen=MAX_SEQ_LEN)
X_test = pad_sequences(seq_test, maxlen=MAX_SEQ_LEN)

# prepare embedding matrix
all_embs = np.stack(embedding_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()

word_index = tokenizer.word_index
num_words = min(MAX_NUM_WORDS, len(word_index))
emb_matrix = np.random.normal(emb_mean, emb_std, (num_words, EMB_DIM))

for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    emb_vector = embedding_index.get(word)
    if emb_vector is not None:
        emb_matrix[i] = emb_vector


def get_model():
    inp = Input(shape=(MAX_SEQ_LEN,))
    x = Embedding(input_dim=num_words, output_dim=EMB_DIM,
                  weights=[emb_matrix], input_length=MAX_SEQ_LEN, trainable=False)(inp)
    x = SpatialDropout1D(0.2)(x)
    z = GlobalMaxPool1D()(x)
    x = GlobalMaxPool1D()(Conv1D(EMB_DIM, 4, activation="relu")(x))
    x = Concatenate()([x, z])
    x = Dropout(0.3)(x)
    # x = Conv1D(filters=256, kernel_size=5, activation='relu')(x)
    # x = MaxPooling1D(5)(x)
    # x = Conv1D(128, 5, activation='relu')(x)
    # x = MaxPooling1D(5)(x)
    # x = Conv1D(64, 5, activation='relu')(x)
    # x = GlobalMaxPool1D()(x)
    # x = Dropout(0.2)(x)
    # x = Dense(50, activation='relu')(x)
    # x = Dropout(0.2)(x)
    preds = Dense(units=y_train.shape[1], activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=preds)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


model = get_model()
batch_size = 128
epochs = 5
file_path = "pretrained_emb_model_3_6.hdf5"
model_checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
stopping = EarlyStopping(monitor='val_loss', mode='min', patience=20)

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
          validation_split=0.1, callbacks=[model_checkpoint, stopping])
model.load_weights(file_path)

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y_test = model.predict(X_test)
print(y_test[0])
submission_df = pd.read_csv("D:/kaggleDatas/Toxic Comment Classification Challenge/sample_submission.csv")
submission_df[list_classes] = y_test
submission_df.to_csv("submission.csv", index=False)
