import numpy as np
import pandas as pd
import tensorflow as tf


def MinMaxScale(data):
    temp1 = data - np.min(data, axis=0)
    temp2 = np.max(data, axis=0) - np.max(data, axis=0)
    return temp1 / (temp2 + 1e-7)


def get_data(is_test):
    if is_test:
        data_df = pd.read_csv("test.csv.zip", compression='zip')
        data = data_df.values[:, 1:]  # ignore ID
        lables = data_df["Id"].values
    else:
        data_df = pd.read_csv("train.csv.zip", compression='zip')
        data = data_df.values[:, 1:-1]  # ignore Id and cover_type
        lables = data_df["Cover_Type"].values
    # print(data_df.head(n=1))
    return lables, data


y_train, x_train = get_data(0)
y_train -= 1
y_train = np.expand_dims(y_train, 1)
train_set_len = len(x_train)

test_id, x_test = get_data(1)
print(x_train.shape, x_test.shape)

x_all = np.vstack((x_train, x_test))
print(x_all.shape)

scale_x = MinMaxScale(x_all)

x_train = scale_x[:train_set_len]
x_test = scale_x[train_set_len:]

print(x_train.shape, x_test.shape)

# set hyper_parameters
learning_rate = 0.1
epoch_nums = 50
step_size = 1000
batch_size = 32
display_freq = 1
n_input = x_train.shape[1]
n_classes = 7
# n_hidden1 = 32

# set placeholder
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.int32, [None, 1])

y_one_hot = tf.one_hot(y, n_classes)
print("y_one_hot shape:", y_one_hot)
y_one_hot = tf.reshape(y_one_hot, [-1, n_classes])

# set weights and biaes
w1 = tf.Variable(tf.random_normal([n_input, n_classes], stddev=0.1))
b1 = tf.Variable(tf.random_normal([n_classes], stddev=0.1))


# w2 = tf.Variable(tf.random_normal([n_hidden1, n_classes], stddev=0.1))
# b2 = tf.Variable(tf.random_normal([n_classes], stddev=0.1))

def model(input_x):
    model_logit = tf.matmul(input_x, w1) + b1
    return model_logit


# set loss and train step
model_logit = model(x)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_logit, labels=y_one_hot))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# set pred op
pred_value = tf.argmax(tf.nn.softmax(model_logit), axis=1)
pred_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_one_hot, 1), pred_value), tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(epoch_nums):
    avg_cost = 0.0
    avg_acc = 0.0
    for step in range(step_size):
        offset = (step * batch_size) % (y_train.shape[0] - batch_size)
        batch_x = x_train[offset: (offset + batch_size), :]
        batch_y = y_train[offset: (offset + batch_size), :]
        _, temp_loss, temp_acc = sess.run([train_step, loss, pred_acc], feed_dict={x: batch_x, y: batch_y})
        avg_acc += temp_acc / step_size
        avg_cost += temp_loss / step_size
    if epoch % display_freq == 0:
        print("epoch %d" % (epoch + 1), "train loss = {:.4f}".format(avg_cost),
              "train accuracy = {:.4f}".format(avg_acc))
print("training finished!!!")

pred_values = sess.run(pred_value, feed_dict={x: x_test})
pred_values += 1

submission = ['Id,Cover_Type']
for i, p in zip(test_id, pred_values):
    submission.append('{0},{1}'.format(i, int(p)))
submission = '\n'.join(submission)

with open("submission2.csv", "w") as outfile:
    outfile.write(submission)
