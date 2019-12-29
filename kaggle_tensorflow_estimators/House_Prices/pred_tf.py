# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2018/7/24 16:02

from __future__ import print_function, absolute_import, division
import numpy as np
import pandas as pd
import tensorflow as tf
import shutil
from scipy.stats import skew
from tensorflow.python.feature_column import feature_column

HEADER_NAMES = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
HEADER_DEFAULTS = [[0], [0.0], [0], [0.0], [0], [0]]
TARGET_NAME = 'SalePrice'

# step1: define categorical features and numeric features. we can get the VOCABULARY of categorical features by
# value_counts( ) function. eg. train_df['OverallQual'].value_counts()
CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY = {'OverallQual': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                             'GarageCars': [0, 1, 2, 3, 4], 'FullBath': [0, 1, 2, 3]}
CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.keys())
NUMERIC_FEATURE_NAMES = ['GrLivArea', 'TotalBsmtSF', 'YearBuilt']


def process_dataframe(dataset_df):
    numeric_feats = dataset_df.dtypes[dataset_df.dtypes != 'object'].index  # 取的是num类型属性名称
    skewed_feats = dataset_df[numeric_feats].apply(lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    print(skewed_feats)
    skewed_feats = skewed_feats.index
    dataset_df[skewed_feats] = np.log1p(dataset_df[skewed_feats])
    print(dataset_df.head())
    # data_df = pd.get_dummies(dataset_df)
    dataset_df = dataset_df.fillna(dataset_df.mean())
    dataset_df['GarageCars'] = dataset_df['GarageCars'].astype(int)
    return dataset_df


def normalize(df):
    result = df.copy()
    nor_columns = ['GrLivArea', 'TotalBsmtSF']
    for feature_name in nor_columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


# step2: Define Data Input Function
def generate_pandas_train_input_fn(train_file_name, batch_size=128):
    raw_df = pd.read_csv(train_file_name)
    x = raw_df.loc[:, HEADER_NAMES]
    x = process_dataframe(x)
    x = normalize(x)
    y = raw_df[TARGET_NAME]
    input_fn = tf.estimator.inputs.pandas_input_fn(x=x, y=y, batch_size=batch_size, num_epochs=None, shuffle=True,
                                                   target_column=TARGET_NAME)
    return input_fn


def generate_pandas_eval_input_fn(eval_file_name, batch_size=128):
    raw_df = pd.read_csv(eval_file_name)
    x = raw_df.loc[:, HEADER_NAMES]
    x = process_dataframe(x)
    y = raw_df[TARGET_NAME]
    input_fn = tf.estimator.inputs.pandas_input_fn(x=x, y=y, batch_size=batch_size, num_epochs=1, shuffle=False,
                                                   target_column=TARGET_NAME)
    return input_fn


def generate_pandas_test_input_fn(test_file_name, batch_size=128):
    raw_df = pd.read_csv(test_file_name)
    x = raw_df.loc[:, HEADER_NAMES]
    x = process_dataframe(x)
    # y = raw_df[TARGET_NAME]
    input_fn = tf.estimator.inputs.pandas_input_fn(x=x, y=None, batch_size=batch_size, shuffle=False,
                                                   target_column=TARGET_NAME)
    return input_fn


# step3: Define Feature Columns
def get_feature_columns():
    raw_numeric_columns = {name: tf.feature_column.numeric_column(key=name) for name in NUMERIC_FEATURE_NAMES}
    # change 'YearBuilt' to a bucketized column
    numeric_columns = {}
    for item in raw_numeric_columns.items():
        if item[0] == 'YearBuilt':
            numeric_columns.update({item[0]: tf.feature_column.bucketized_column(item[1],
                                                                                 boundaries=[1900, 1920, 1940, 1960,
                                                                                             1980, 2000])})
        else:
            numeric_columns.update({item[0]: item[1]})

    cate_columns = {
        item[0]: tf.feature_column.categorical_column_with_vocabulary_list(key=item[0], vocabulary_list=item[1])
        for item in CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.items()}
    feature_columns = {}
    if numeric_columns is not None:
        feature_columns.update(numeric_columns)
    if cate_columns is not None:
        feature_columns.update(cate_columns)
    return feature_columns


# step4: Create an Estimator
def create_estimator(model_dir):
    feature_columns = list(get_feature_columns().values())
    # get dense features
    dense_features = list(filter(lambda column: isinstance(column, feature_column._NumericColumn), feature_columns))
    # change cate features and bucketized features to indicator features
    cate_features = list(filter(
        lambda column: isinstance(column, feature_column._VocabularyListCategoricalColumn) |
                       isinstance(column, feature_column._BucketizedColumn), feature_columns))
    indicator_features = list(map(lambda column: tf.feature_column.indicator_column(column), cate_features))
    final_feature_columns = dense_features + indicator_features
    dnn_estimator = tf.estimator.DNNRegressor(feature_columns=final_feature_columns, hidden_units=[100, 50],
                                              model_dir=model_dir, optimizer=tf.train.AdamOptimizer())
    return dnn_estimator


# step5: Train and eval the Estimator
def train_and_eval(train_file, eval_file, save_model_path):
    shutil.rmtree(save_model_path, ignore_errors=True)
    dnn_model = create_estimator(save_model_path)
    train_spec = tf.estimator.TrainSpec(input_fn=generate_pandas_train_input_fn(train_file), max_steps=20000)
    # we do evaluation on training data
    eval_spec = tf.estimator.EvalSpec(input_fn=generate_pandas_eval_input_fn(eval_file))
    tf.estimator.train_and_evaluate(estimator=dnn_model, train_spec=train_spec, eval_spec=eval_spec)


# step6: get prediction values in test data
def test(test_file, save_model_path):
    dnn_model = create_estimator(save_model_path)
    predictions = dnn_model.predict(input_fn=generate_pandas_test_input_fn(test_file))
    print(predictions)
    preds = [p["predictions"][0] for p in predictions]
    print(len(preds))
    ids = pd.read_csv(test_file).Id
    submission = ['Id,SalePrice']
    for id, prediction in zip(ids, preds):
        submission.append('{0},{1}'.format(id, prediction))
    submission = '\n'.join(submission)
    with open('submission.csv', 'w') as outfile:
        outfile.write(submission)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    train_and_eval("train.csv", "train.csv", save_model_path="dnn_pre_estimator_model")
    test("test.csv", save_model_path="dnn_pre_estimator_model")


if __name__ == '__main__':
    # tf.app.run()
    df = pd.read_csv("train.csv")
    generate_pandas_train_input_fn("train.csv")