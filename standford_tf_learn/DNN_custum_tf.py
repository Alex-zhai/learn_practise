import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import requests
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

birth_weight_file = 'birth_weight.csv'

# download data and create data file if file does not exist in current directory
if not os.path.exists(birth_weight_file):
    birthdata_url = 'https://github.com/nfmcclure/tensorflow_cookbook/raw/master/01_Introduction/' \
                    '07_Working_with_Data_Sources/birthweight_data/birthweight.dat'
    birth_file = requests.get(birthdata_url)
    birth_data = birth_file.text.split('\r\n')
    birth_header = birth_data[0].split('\t')
    birth_data = [[float(x) for x in y.split('\t') if len(x) >= 1] for y in birth_data[1:] if len(y) >= 1]
    with open(birth_weight_file, "w") as f:
        writer = csv.writer(f)
        writer.writerows([birth_header])
        writer.writerows(birth_data)
        f.close()

data_df = pd.read_csv('birth_weight.csv')
scaler = MinMaxScaler()
data_df.iloc[:, 0:-1] = scaler.fit_transform(data_df.iloc[:, 0:-1])
print(data_df.head())
data_size = data_df.shape[0]
cols = data_df.columns.values

data_df.to_csv("processed_birth_weight_data.csv", header=False, index=False)
_csv_columns_defaults = []
for i in range(len(cols)):
    _csv_columns_defaults.append([1.0])


def my_model(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    model_logit = tf.squeeze(tf.layers.dense(net, 1, activation=None), 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'prediction': model_logit
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    loss = tf.losses.mean_squared_error(labels=labels, predictions=model_logit)
    train_op = tf.train.AdamOptimizer(learning_rate=0.01).\
        minimize(loss, global_step=tf.train.get_global_step())
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, train_op=train_op, loss=loss)
    assert mode == tf.estimator.ModeKeys.EVAL
    rmse = tf.metrics.mean_squared_error(labels=labels, predictions=model_logit)
    metrics = {'rmse': rmse}
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)


def input_fn(data_file, num_epochs, shuffle, batch_size):
    assert tf.gfile.Exists(data_file)

    def parse_csv(value):
        print('Parsing', data_file)
        columns = tf.decode_csv(value, record_defaults=_csv_columns_defaults)
        features = dict(zip(cols, columns))
        labels = features.pop('BWT')
        return features, labels
    dataset = tf.data.TextLineDataset(data_file)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=data_size)
    dataset = dataset.map(parse_csv, num_parallel_calls=5)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


def train():
    feat_cols = [tf.feature_column.numeric_column(key=feat) for feat in cols[:-1]]
    model = tf.estimator.Estimator(
        model_fn=my_model, model_dir='dnn_model',
        params={
            'feature_columns': feat_cols,
            'hidden_units': [25, 10, 5]
        }
    )
    for i in range(10):
        model.train(input_fn=lambda: input_fn('processed_birth_weight_data.csv', 2, True, 50))
        results = model.evaluate(input_fn=lambda: input_fn("processed_birth_weight_data.csv", 1, False, 50))
        print("Result at epoch", i)
        print('-' * 60)
        for key in sorted(results):
            print("%s: %s" % (key, results[key]))

if __name__ == '__main__':
    train()

