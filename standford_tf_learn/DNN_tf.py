import tensorflow as tf
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data_df = pd.read_csv('processed_birth_weight_data.csv')

feature_data = data_df.iloc[:, :-1].values
labels = data_df.iloc[:, -1].values
print(feature_data.shape, labels.shape)

X_train, X_test, y_train, y_test = train_test_split(feature_data, labels, test_size=0.2, random_state=42)
print(X_train)

feat_columns = [tf.feature_column.numeric_column(key='x', shape=np.array(X_train).shape[1:])]

reg_model = tf.estimator.DNNClassifier(
    feature_columns=feat_columns, hidden_units=[10, 10]
)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': X_train}, y=y_train, batch_size=1, num_epochs=None, shuffle=True
)
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': X_test}, y=y_test, batch_size=1, num_epochs=1, shuffle=False
)
for i in range(50):
    reg_model.train(input_fn=train_input_fn, steps=1000)
    score_tf = reg_model.evaluate(input_fn=test_input_fn)
    print("tf MSE is {}".format(score_tf))

# preds = reg_model.predict(input_fn=test_input_fn)
#
# pred_values = np.array(list(pred['predictions'] for pred in preds))
#
# score_sk = mean_squared_error(y_true=y_test, y_pred=pred_values)
# print("sk MSE is {}".format(score_sk))
#
# score_tf = reg_model.evaluate(input_fn=test_input_fn)
# print("tf MSE is {}".format(score_tf))