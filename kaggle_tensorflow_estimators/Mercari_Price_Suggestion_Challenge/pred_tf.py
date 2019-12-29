# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/7/27 15:03

from __future__ import division, absolute_import, print_function
import tensorflow as tf
import pandas as pd
import numpy as np
import shutil
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import layers, models, preprocessing

MAX_NAME_SEQ = 10
MAX_DEC_SEQ = 100

# step1: prepare data
train_df = pd.read_table("D:/kaggleDatas/Mercari Price Suggestion Challenge/train.tsv")
test_df = pd.read_table("D:/kaggleDatas/Mercari Price Suggestion Challenge/test.tsv")


# handle missing values
def fill_missing_data(pd_df):
    pd_df['category_name'].fillna(value='missing', inplace=True)
    pd_df['brand_name'].fillna(value='missing', inplace=True)
    pd_df['item_description'].fillna(value='missing', inplace=True)
    return pd_df


train_df = fill_missing_data(train_df[:100])
test_df = fill_missing_data(test_df[:20])


# handle categorical features
def handle_cate_features(df_train, df_test):
    le = LabelEncoder()
    le.fit(list(df_train['category_name']) + list(df_test['category_name']))
    df_train['category_name'] = le.transform(df_train['category_name'])
    df_test['category_name'] = le.transform(df_test['category_name'])

    le.fit(list(df_train['brand_name']) + list(df_test['brand_name']))
    df_train['brand_name'] = le.transform(df_train['brand_name'])
    df_test['brand_name'] = le.transform(df_test['brand_name'])
    return df_train, df_test


train_df, test_df = handle_cate_features(train_df, test_df)


# handle text features
def handel_text_features(df_train, df_test):
    raw_text = list(df_train['name']) + list(df_test['name']) + list(df_train['item_description']) + list(
        df_test['item_description'])
    tokenizer = preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(raw_text)
    token_nums = len(tokenizer.word_index) + 1
    df_train['seq_name'] = tokenizer.texts_to_sequences(df_train['name'].str.lower())
    df_test['seq_name'] = tokenizer.texts_to_sequences(df_test['name'].str.lower())
    df_train['seq_item_description'] = tokenizer.texts_to_sequences(df_train['item_description'].str.lower())
    df_test['seq_item_description'] = tokenizer.texts_to_sequences(df_test['item_description'].str.lower())
    return token_nums, df_train, df_test


token_nums, train_df, test_df = handel_text_features(train_df, test_df)

print(train_df.head(5))


# print(train_df.columns.values)
# print(train_df['item_description'].values)


# step2: split train data to train ane val data
def get_train_and_eval_data(df_train):
    train_feature_dict = {
        'pad_seq_name': preprocessing.sequence.pad_sequences(df_train['seq_name'], maxlen=MAX_NAME_SEQ,
                                                             padding='post', value=0),
        'pad_seq_item_description': preprocessing.sequence.pad_sequences(df_train['seq_item_description'],
                                                                         maxlen=MAX_DEC_SEQ, padding='post', value=0),
        'item_condition_id': df_train['item_condition_id'].values, 'category_name': df_train['category_name'].values,
        'brand_name': df_train['brand_name'].values, 'shipping': df_train['shipping'].values}
    train_y = np.log(df_train['price'] + 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_y = scaler.fit_transform(train_y.reshape(-1, 1))
    train_X, valid_X, train_y, valid_y = train_test_split(train_x, train_y, random_state=42, test_size=0.1)
    return train_X, valid_X, train_y, valid_y


# get test data for prediction
def get_test_data(df_test):
    df_test['pad_seq_name'] = preprocessing.sequence.pad_sequences(df_test['seq_name'].values, maxlen=MAX_NAME_SEQ,
                                                                   padding='post', value=0)

    df_test['pad_seq_item_description'] = preprocessing.sequence.pad_sequences(df_test['seq_item_description'].values,
                                                                               maxlen=MAX_DEC_SEQ, padding='post',
                                                                               value=0)
    return df_test.values


train_X, valid_X, train_y, valid_y = get_train_and_eval_data(train_df)
test_X = get_test_data(test_df)
print(train_X[0])


# step3: prepare input function
def train_input_fn(batch_size=128):
    dataset = tf.data.Dataset.from_tensor_slices(({"name": train_X}, train_y))
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
