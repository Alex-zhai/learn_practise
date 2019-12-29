from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=42)

# create tf model
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
feat_cols = [tf.feature_column.numeric_column(key='x', shape=np.array(X_train).shape[1:])]

model = tf.estimator.DNNClassifier(hidden_units=[10, 20, 10], feature_columns=feat_cols, n_classes=3)
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': X_train}, y=y_train, num_epochs=None, shuffle=True)

model.train(input_fn=train_input_fn, steps=200)

test_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': X_test}, y=y_test, num_epochs=1, shuffle=False)
predictions = model.predict(input_fn=test_input_fn)
y_preds = list(p['class_ids'][0] for p in predictions)
print(y_preds)
print(accuracy_score(y_test, y_preds))














# Data sets
# IRIS_TRAINING = 'iris_training.csv'
# IRIS_TRAINING_URL = 'http://download.tensorflow.org/data/iris_training.csv'
#
# IRIS_TEST = 'iris_test.csv'
# IRIS_TEST_URL = 'http://download.tensorflow.org/data/iris_test.csv'
#
# FEATURE_KEYS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
#
#
# def maybe_download_iris_data(file_name, download_url):
#     if not os.path.exists(file_name):
#         raw = urllib.urlopen(download_url).read()
#         with open(file_name, 'w') as f:
#             f.write(raw)
#     with open(file_name, "r") as f_read:
#         first_line = f_read.readline()
#     num_elements = first_line.split(',')[0]
#     print(num_elements)
#     return int(num_elements)
#
#
# def input_fn(file_name, num_data, batch_size, is_training):
#     """Creates an input_fn required by Estimator train/evaluate."""
#     # If the data sets aren't stored locally, download them.
#     def _parse_csv(value):
#         print("Parsing", file_name)
#         columns = tf.decode_csv(value, record_defaults=[[]]*(len(FEATURE_KEYS)+1))
#         features = dict(zip(FEATURE_KEYS, columns[:len(FEATURE_KEYS)]))
#         labels = tf.cast(columns[len(FEATURE_KEYS)+1], tf.int32)
#         return features, labels
#
#     dataset = tf.data.TextLineDataset([file_name])
#     dataset = dataset.skip(1)
#     dataset = dataset.map(_parse_csv)
#     if is_training:
#         dataset = dataset.shuffle(num_data)
#         dataset = dataset.repeat()
#     dataset = dataset.batch(batch_size)
#     iterator = dataset.make_one_shot_iterator()
#     features, labels = iterator.get_next()
#     return features, labels
#
#
# if __name__ == '__main__':
#     num_training_data = maybe_download_iris_data(IRIS_TRAINING, IRIS_TRAINING_URL)
#     num_test_data = maybe_download_iris_data(IRIS_TEST, IRIS_TEST_URL)
#
#     # Build 3 layer DNN with 10, 20, 10 units respectively.
#     feature_cols = [tf.feature_column.numeric_column(key) for key in FEATURE_KEYS]
#     model = tf.estimator.DNNClassifier(hidden_units=(10, 20, 10), feature_columns=feature_cols, n_classes=3)
#
#     train_input_fn = input_fn(IRIS_TRAINING, num_training_data, batch_size=32, is_training=True)
#     model.train(input_fn=train_input_fn, steps=400)
#
#     test_input_fn = input_fn(IRIS_TEST, num_test_data, batch_size=32, is_training=False)
#     scores = model.evaluate(input_fn=test_input_fn)
#     print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))
