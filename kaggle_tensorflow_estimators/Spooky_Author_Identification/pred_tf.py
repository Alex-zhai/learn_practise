# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/7/26 16:30

from __future__ import division, absolute_import, print_function
import tensorflow as tf
import pandas as pd
import numpy as np
import shutil
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import layers, models, preprocessing

MAX_NUM_WORDS = 30000
EMB_DIM = 300
MAX_SEQ_LEN = 70

# prepare data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

all_data_x = list(train_df['text'].values) + list(test_df['text'].values)


# split train data to train ane val data
def get_train_and_eval_data(train_df):
    train_x = train_df['text'].values
    train_y = train_df['author'].values
    tokenizer = preprocessing.text.Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(all_data_x)
    word_index_len = len(tokenizer.word_index)
    train_x_seqs = tokenizer.texts_to_sequences(train_x)
    # print(max(len(sep) for sep in train_x_seqs))
    train_x_pad_seqs = preprocessing.sequence.pad_sequences(train_x_seqs, maxlen=MAX_SEQ_LEN, padding='post', value=0)
    label_encoder = LabelEncoder()
    en_train_y = label_encoder.fit_transform(train_y)
    train_X, valid_X, train_y, valid_y = train_test_split(train_x_pad_seqs, en_train_y, random_state=42, test_size=0.1)
    return word_index_len, train_X, valid_X, train_y, valid_y


# get test data for prediction
def get_test_data(test_df):
    test_x = test_df['text'].values
    tokenizer = preprocessing.text.Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(all_data_x)
    test_x_seqs = tokenizer.texts_to_sequences(test_x)
    test_x_pad_seqs = preprocessing.sequence.pad_sequences(test_x_seqs, maxlen=MAX_SEQ_LEN, padding='post', value=0)
    return test_x_pad_seqs


# prepare input function
word_index_len, train_X, valid_X, train_y, valid_y = get_train_and_eval_data(train_df)
test_X = get_test_data(test_df)


def train_input_fn(batch_size=128):
    dataset = tf.data.Dataset.from_tensor_slices(({"text": train_X}, train_y))
    dataset = dataset.shuffle(2 * batch_size + 1)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def eval_input_fn(batch_size=128):
    dataset = tf.data.Dataset.from_tensor_slices(({"text": valid_X}, valid_y))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def test_input_fn(batch_size=128):
    dataset = tf.data.Dataset.from_tensor_slices(({"text": test_X}))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


# create model
def create_gru_model(input_shape=(MAX_SEQ_LEN,)):
    input_layer = layers.Input(shape=input_shape)
    x = layers.Embedding(input_dim=min(word_index_len, MAX_NUM_WORDS) + 1, output_dim=300,
                         input_length=MAX_SEQ_LEN)(input_layer)
    x = layers.SpatialDropout1D(0.1)(x)
    x = layers.GRU(300, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)(x)
    x = layers.GRU(300, dropout=0.3, recurrent_dropout=0.3)(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.8)(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.8)(x)
    logits = layers.Dense(3)(x)
    return models.Model(inputs=input_layer, outputs=logits)


def gru_model_fn(features, labels, mode, params):
    features = features["text"]
    # labels = tf.cast(labels, tf.int32)
    model = create_gru_model((MAX_SEQ_LEN,))
    logits = model(features)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, axis=1),
        }
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions, export_outputs={
            'predict': tf.estimator.export.PredictOutput(predictions)
        })

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN, loss=loss,
            train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step())
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        eval_metric_spec = {
            'accuracy': tf.metrics.accuracy(labels, predictions=tf.argmax(input=logits, axis=1)),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL, loss=loss, eval_metric_ops=eval_metric_spec
        )


def train_and_eval(save_model_path):
    shutil.rmtree(save_model_path, ignore_errors=True)
    gru_model = tf.estimator.Estimator(
        model_fn=gru_model_fn, model_dir=save_model_path, params={
            'learning_rate': 0.001,
        }
    )
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(), max_steps=20000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn())
    tf.estimator.train_and_evaluate(estimator=gru_model, train_spec=train_spec, eval_spec=eval_spec)


#  get prediction values in test data
def test(submit_file, save_model_path):
    gru_model = tf.estimator.Estimator(
        model_fn=gru_model_fn, model_dir=save_model_path, params={
            'learning_rate': 0.001,
        }
    )
    predictions = gru_model.predict(input_fn=lambda: test_input_fn())
    preds = [list(p["probabilities"]) for p in predictions]
    # print(len(preds))
    preds = np.asarray(preds)
    submission_df = pd.read_csv(submit_file)
    class_names = submission_df.columns.values[1:]
    submission_df[class_names] = preds
    submission_df.to_csv("gru_tf_est_submission.csv", index=False)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    train_and_eval("gru_model")
    test("sample_submission.csv", "gru_model")


if __name__ == '__main__':
    # model = create_gru_model()
    # print(model.summary())
    tf.app.run()
