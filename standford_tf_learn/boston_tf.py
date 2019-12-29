import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn import metrics
from sklearn import model_selection

# Load dataset
boston = datasets.load_boston()
x, y = boston.data, boston.target

# Split dataset into train / test
train_x, test_x, train_y, test_y = model_selection.train_test_split(x, y, test_size=0.2, random_state=42)

# Scale data (training set) to 0 mean and unit standard deviation.
scaler = preprocessing.StandardScaler()
train_x = scaler.fit_transform(train_x)

# Build 2 layer fully connected DNN with 10, 10 units respectively.
feature_columns = [tf.feature_column.numeric_column(key='x', shape=np.array(train_x).shape[1:])]
reg_model = tf.estimator.DNNClassifier(hidden_units=[10,10], feature_columns=feature_columns)

# Train.
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': train_x}, y=train_y,
                                                    batch_size=1, num_epochs=None, shuffle=True)
reg_model.train(input_fn=train_input_fn, steps=2000)
# Predict.
test_x = scaler.transform(test_x)
test_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': test_x}, y=test_y,
                                                   num_epochs=1, shuffle=False)
predictions = reg_model.predict(input_fn=test_input_fn)
y_preds = np.array(list(p["predictions"] for p in predictions))
y_preds = y_preds.reshape(np.array(test_y).shape)
# Score with sklearn.
sk_score = metrics.mean_squared_error(y_true=test_y, y_pred=y_preds)
print('MSE of sklearn is {0:f}'.format(sk_score))
# Score with tensorflow.
tf_score = reg_model.evaluate(input_fn=test_input_fn)
print('MSE of tf is {0:f}'.format(tf_score['average_loss']))