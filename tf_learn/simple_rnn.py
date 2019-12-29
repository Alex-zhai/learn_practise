# coding = utf-8
# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# set hyper-parameters
input_size = 28
time_step = 28
batch_size = 64
num_hidden = 128
num_class = 10

learning_rate = 0.001
dis_freq = 200
train_steps = 10000

# set input placeholder
x = tf.placeholder("float", shape=[None, time_step, input_size])
y = tf.placeholder("float", shape=[None, num_class])

sess = tf.Session()
# set variable
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_class]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_class]))
}

# set model
def simple_rnn(input_x, weights, biases):
    # input_x = tf.transpose(input_x, [1, 0, 2])
    input_x = tf.unstack(x, time_step, 1) #
    lstm_cell = rnn.BasicLSTMCell(num_units=num_hidden)
    outputs, states = rnn.static_rnn(lstm_cell, input_x, dtype=tf.float32)
    # outputs (?, 128)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

model_output = simple_rnn(x, weights, biases)
pred = tf.nn.softmax(model_output)
# set loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_output, labels=y))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# get accuarcy
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), tf.float32))


init = sess.run(tf.global_variables_initializer())

for i in range(train_steps):
    # get batch data
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    # reshape to [none, time_step, input_size]
    batch_x = batch_x.reshape((batch_size, time_step, input_size))
    sess.run(train_step, feed_dict={x: batch_x, y: batch_y})
    if (i+1) % dis_freq == 0 or i==0:
        tmp_loss, tmp_acc = sess.run([loss, accuracy], feed_dict={x: batch_x, y: batch_y})
        # 注意取名字的时候不能取和action相同名字
        print("Step " + str(i + 1) + ", minibatch loss= " + "{:.4f}".format(tmp_loss) + ", Training Accuracy=" + \
              "{:.3f}".format(tmp_acc))
print("Optimization Finished!")

# test
test_len = 128
test_data = mnist.test.images[:test_len].reshape((-1, time_step, input_size))
test_label = mnist.test.labels[:test_len]
print("testing accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))