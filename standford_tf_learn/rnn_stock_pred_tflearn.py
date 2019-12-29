import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# set hyper_parameters
input_nums = 5
seq_length = 7
num_classes = 1
hidden_size = 10
learning_rate = 0.01
train_steps = 500

data = np.loadtxt("data-02-stock_daily.csv", delimiter=",")
print(data.shape)
scaler = MinMaxScaler()
scaler_data = scaler.fit_transform(data)

feature_data = scaler_data[::-1]
print(feature_data.shape)
label = scaler_data[:, [-1]]

x_data = []
y_data = []
for i in range(len(label) - seq_length):
    x_data.append(feature_data[i:i+seq_length])
    y_data.append(label[i+seq_length])

x_data = np.array(x_data)
y_data = np.array(y_data)
print(x_data.shape, y_data.shape)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# set placeholder
x = tf.placeholder(dtype=tf.float32, shape=[None, seq_length, input_nums])
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

# set model
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True,
                                        activation=tf.nn.tanh)
# [batch_size, max_time, cell.output_size]
outputs, state = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=x, dtype=tf.float32)  #shape=(?, 7, 10)

outputs = outputs[:, -1]
# set variables
w = tf.get_variable(name='w', shape=[hidden_size, num_classes])
b = tf.get_variable(name='b', shape=[num_classes])
model_logits = tf.matmul(outputs, w) + b

# set loss function and train_op
loss = tf.losses.mean_squared_error(labels=y, predictions=model_logits)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(train_steps):
    _, temp_loss = sess.run([train_op, loss], feed_dict={x: X_train, y: y_train})
    print("step %d" % (i+1), "loss is {:.9f}".format(temp_loss))

pred = sess.run(model_logits, feed_dict={x: X_test})
print("Rmse is {}".format(mean_squared_error(y_true=y_test, y_pred=pred)))

plt.plot(y_test)
plt.plot(pred)
plt.xlabel('time period')
plt.ylabel('stock price')
plt.show()