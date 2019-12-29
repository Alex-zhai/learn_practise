import tensorflow as tf
import numpy as np
import pandas as pd


def MinMaxScale(data):
    temp1 = data - np.min(data, axis=0)
    temp2 = np.max(data, axis=0) - np.min(data, axis=0)
    return temp1 / (temp2 + 1e-7)


def load_data(is_test):
    if is_test:
        data_df = pd.read_csv("test.csv")
        data = data_df.values[:, 1:]
        labels = data_df["datetime"].values
    else:
        data_df = pd.read_csv("train.csv")
        data = data_df.values[:, 1:-3]
        labels = data_df["count"].values
    return data, labels


x_train, y_train = load_data(0)
print(x_train.shape, y_train.shape)
y_train -= 1
y_train = np.expand_dims(y_train, axis=1)
train_len = len(x_train)

x_test, test_id = load_data(1)
print(x_test.shape)
x_all = np.vstack((x_train, x_test))
print(x_all.shape)

x_all_scale = MinMaxScale(x_all)

x_train = x_all_scale[:train_len]
x_test = x_all_scale[train_len:]

print(x_train.shape, x_test.shape)

# set hyper-parameters
learning_rate = 0.1
batch_size = 32
epoch_nums = 100
display_freq = 10
input_num = x_train.shape[1]

# set placeholder
x = tf.placeholder(tf.float32, [None, input_num])
y = tf.placeholder(tf.float32, [None, 1])

# set Varible
w = tf.Variable(tf.random_normal([input_num, 1]))
b = tf.Variable(tf.random_normal([1]))

# set model
model_logit = tf.matmul(x, w) + b

# set loss and train step
loss = tf.reduce_mean(tf.square(y - model_logit))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(epoch_nums):
    avg_cost = 0.0
    steps = int(x_train.shape[0] / batch_size) + 1
    for step in range(steps):
        offset = (step * batch_size) % (y_train.shape[0] - batch_size)
        batch_x = x_train[offset: (offset + batch_size), :]
        batch_y = y_train[offset: (offset + batch_size), :]
        _, temp_loss = sess.run([train_step, loss], feed_dict={x: batch_x, y: batch_y})
        avg_cost += temp_loss / steps
    if epoch % display_freq == 0:
        print("Epoch: %02d" % (epoch + 1), "cost = {:.4f}".format(avg_cost))
print("training finished!!!")

preds = sess.run(model_logit, feed_dict={x: x_test})
submission = ['datetime,count']
for id, pred in zip(test_id, preds):
    if int(pred) < 0:
        pred = 0
    submission.append('{0},{1}'.format(id, int(pred)))
submission = '\n'.join(submission)

with open("submission_tf.csv", "w") as outfile:
    outfile.write(submission)
