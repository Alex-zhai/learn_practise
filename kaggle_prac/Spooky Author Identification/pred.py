from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import xgboost as xgb
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, LSTM, Dense, Dropout, SpatialDropout1D, GRU, Embedding
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

train_df = pd.read_csv("D:/kaggleDatas/Spooky Author Identification/train.csv")
test_df = pd.read_csv("D:/kaggleDatas/Spooky Author Identification/test.csv")


def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param eps:
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota


label_encoder = LabelEncoder()
train_y = label_encoder.fit_transform(train_df['author'].values)
train_X, valid_X, train_y, valid_y = train_test_split(train_df['text'].values, train_y, stratify=train_y,
                                                      random_state=42, test_size=0.1)

print(train_X.shape, valid_X.shape)

#
# # TF-IDF model
tfidf_vectorizer = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word',
                                   token_pattern=r'\w{1,}', ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                                   stop_words='english')
tfidf_vectorizer.fit(list(train_X) + list(valid_X))
tfv_trainX = tfidf_vectorizer.transform(train_X)
tfv_validX = tfidf_vectorizer.transform(valid_X)

lr = LogisticRegression(C=1.0)
lr.fit(tfv_trainX, train_y)
predict_proba = lr.predict_proba(tfv_validX)
print("The logloss of lr model using tfidf is: %0.3f" % multiclass_logloss(valid_y, predict_proba))

# count TF
count_vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 3), stop_words='english')
count_vectorizer.fit(list(train_X) + list(valid_X))
ctv_trainX = count_vectorizer.transform(train_X)
ctv_validX = count_vectorizer.transform(valid_X)

lr = LogisticRegression(C=1.0)
lr.fit(ctv_trainX, train_y)
predict_proba = lr.predict_proba(ctv_validX)
print("The logloss of lr model using count TF is: %0.3f" % multiclass_logloss(valid_y, predict_proba))

# nb model
nb = MultinomialNB()
nb.fit(tfv_trainX, train_y)
nb_predict_proba = nb.predict_proba(tfv_validX)
print("The logloss of nb model using count tfidf is: %0.3f" % multiclass_logloss(valid_y, nb_predict_proba))

# svm model
svd = TruncatedSVD(n_components=120)
svd.fit(tfv_trainX)
svd_trainX = svd.transform(tfv_trainX)
svd_validX = svd.transform(tfv_validX)

standard_scaler = StandardScaler()
standard_scaler.fit(svd_trainX)
scaler_svd_trainX = standard_scaler.transform(svd_trainX)
scaler_svd_validX = standard_scaler.transform(svd_validX)

svc = SVC(C=1.0, probability=True)
svc.fit(scaler_svd_trainX, train_y)
svc_predict_proba = svc.predict_proba(scaler_svd_validX)
print("The logloss of svc model using count tfidf is: %0.3f" % multiclass_logloss(valid_y, svc_predict_proba))

xgb_classifier = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, subsample=0.8, nthread=10,
                                   learning_rate=0.1)
xgb_classifier.fit(tfv_trainX.tocsc(), train_y)
xgb_predict_proba = xgb_classifier.predict_proba(tfv_validX.tocsc())
print("The logloss of xgb model using count tfidf is: %0.3f" % multiclass_logloss(valid_y, xgb_predict_proba))

scorer = make_scorer(multiclass_logloss, greater_is_better=False, needs_proba=True)

# pipeline
svd = TruncatedSVD()
scaler = StandardScaler()
lr = LogisticRegression()
pipeline_model = Pipeline([('svd', svd), ('scaler', scaler), ('lr', lr)])

param_grid = {
    'svd__n_components': [120, 180],
    'lr__C': [0.1, 1.0, 10],
    'lr__penalty': ['l1', 'l2']
}

cv_model = GridSearchCV(estimator=pipeline_model, param_grid=param_grid, scoring=scorer, n_jobs=-1, verbose=10, cv=2)
cv_model.fit(tfv_trainX, train_y)
print("best score: %3f" % cv_model.best_score_)
print("best parameters set:")
best_parameters = cv_model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# deep learning
# lstm model
MAX_NUM_WORDS = 30000
EMB_DIM = 300
MAX_SEQ_LEN = 70


def get_padseq_data(x, fit_on_data, num_words, max_seq_len):
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(fit_on_data)
    seq_X = tokenizer.texts_to_sequences(x)
    padseq_X = pad_sequences(seq_X, maxlen=max_seq_len)
    word_index = tokenizer.word_index
    return word_index, padseq_X


fit_on_data = list(train_X) + list(valid_X)
word_index, padseq_trainX = get_padseq_data(train_X, fit_on_data, MAX_NUM_WORDS, MAX_SEQ_LEN)
_, padseq_validX = get_padseq_data(valid_X, fit_on_data, MAX_NUM_WORDS, MAX_SEQ_LEN)
_, padseq_testX = get_padseq_data(test_df['text'].values, fit_on_data, MAX_NUM_WORDS, MAX_SEQ_LEN)

cat_train_y = to_categorical(train_y)
cat_valid_y = to_categorical(valid_y)

embedding_index = {}
emb_file = open("D:/kaggleDatas/Toxic Comment Classification Challenge/glove.6B.300d.txt", 'r', encoding='utf-8')
for line in emb_file:
    values = line.split()
    word = values[0]
    wordvec = np.asarray(values[1:], dtype=np.float32)
    embedding_index[word] = wordvec
emb_file.close()

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index))
emb_matrix = np.zeros((num_words + 1, EMB_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    emb_vector = embedding_index.get(word)
    if emb_vector is not None:
        emb_matrix[i] = emb_vector


def lstm_model():
    inp = Input(shape=(MAX_SEQ_LEN,))
    x = Embedding(input_dim=num_words + 1, output_dim=EMB_DIM, weights=[emb_matrix], input_length=MAX_SEQ_LEN,
                  trainable=False)(inp)
    x = SpatialDropout1D(0.3)(x)
    x = LSTM(100, dropout=0.3, recurrent_dropout=0.3)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.8)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.8)(x)
    out = Dense(3, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def gru_model():
    inp = Input(shape=(MAX_SEQ_LEN,))
    x = Embedding(input_dim=num_words + 1, output_dim=EMB_DIM, weights=[emb_matrix], input_length=MAX_SEQ_LEN,
                  trainable=False)(inp)
    x = SpatialDropout1D(0.3)(x)
    x = GRU(300, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)(x)
    x = GRU(300, dropout=0.3, recurrent_dropout=0.3)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.8)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.8)(x)
    out = Dense(3, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_callbacks(filepath, patience=3):
    stopping = EarlyStopping(monitor='val_loss', patience=patience, mode='min')
    checkpoint = ModelCheckpoint(filepath=filepath, save_best_only=True)
    return [stopping, checkpoint]


callbacks = get_callbacks("pretrained_lstm_keras.hdf5")
lstm_model = lstm_model()
lstm_model.fit(padseq_trainX, cat_train_y, batch_size=512, epochs=100, verbose=1, callbacks=callbacks,
               validation_data=[padseq_validX, cat_valid_y])

print("*****************************************************************")
callbacks = get_callbacks("pretrained_gru_keras.hdf5")
gru_model = gru_model()
gru_model.fit(padseq_trainX, cat_train_y, batch_size=512, epochs=100, verbose=1, callbacks=callbacks,
              validation_data=[padseq_validX, cat_valid_y])

# test
preds = gru_model.predict(padseq_testX)
submission_df = pd.read_csv("D:/kaggleDatas/Spooky Author Identification/sample_submission.csv")
class_names = submission_df.columns.values[1:]
submission_df[class_names] = preds
submission_df.to_csv("gru_keras_submission.csv", index=False)