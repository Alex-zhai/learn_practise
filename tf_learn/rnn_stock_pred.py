import tensorflow as tf
import numpy as np
import matplotlib
import os

# if "DISPLAY" not in os.environ:
#     matplotlib.use('Agg')

import matplotlib.pyplot as plt

# feature scala
def max_min_scala(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

# set hyper_parameters
input_nums = 5
seq_length = 7
num_classes = 1
hidden_size = 10
learning_rate = 0.01
train_steps = 500

# prepare train and test data
data = np.loadtxt("data-02-stock_daily.csv", delimiter=",")
data = data[::-1]
scala_data = max_min_scala(data)
feature_data = scala_data
print(feature_data.shape)  # 732*5
label_data = scala_data[:, [-1]]


x_data = []
y_data = []

for i in range(len(label_data) - seq_length):
    temp_x = feature_data[i: i+seq_length]
    temp_y = label_data[i + seq_length]
    #print(temp_x, "=>", temp_y)
    x_data.append(temp_x)
    y_data.append(temp_y)

print(len(x_data))
# train and test split
train_x, test_x = np.array(x_data[0: int(0.7*len(y_data))]), np.array(x_data[int(0.7*len(y_data)):len(y_data)])
train_y, test_y = np.array(y_data[0: int(0.7*len(y_data))]), np.array(y_data[int(0.7*len(y_data)):len(y_data)])

print(train_x.shape) # (507, 7, 5)

# set placeholder
x = tf.placeholder(tf.float32, [None, seq_length, input_nums])
y = tf.placeholder(tf.float32, [None, 1])

# set rnn model
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True, activation=tf.tanh)
outputs, _state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
print(outputs)  # shape=(?, 7, 10)
w = tf.get_variable("w", [hidden_size, num_classes])
b = tf.get_variable("b", [num_classes])
print(outputs[:, -1])   # shape=(?, 10)
y_pred = tf.matmul(outputs[:, -1], w) + b
print(y_pred)
# set loss function
loss = tf.reduce_sum(tf.square(y_pred - y))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# get rmse
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(train_steps):
    _, tmp_loss = sess.run([train_step, loss], feed_dict={x: train_x, y: train_y})
    print("step: %d" % (i+1), "loss = {:.9f}".format(tmp_loss))

test_pred = sess.run(y_pred, feed_dict={x: test_x})
rmse_val = sess.run(rmse, feed_dict={targets: test_y, predictions: test_pred})
print("RMSE: {}".format(rmse_val))

plt.plot(test_y)
plt.plot(test_pred)
plt.xlabel('time period')
plt.ylabel('stock price')
plt.show()