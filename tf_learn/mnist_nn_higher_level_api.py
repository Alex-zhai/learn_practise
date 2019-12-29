import tensorflow as tf
import random
from tensorflow.contrib.layers import fully_connected, batch_norm, dropout
from tensorflow.contrib.framework import arg_scope
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

batch_size = 128
learning_rate = 0.001
epoches = 30
keep_prop = 0.7

#set placeholder
x = tf.placeholder(tf.float32, [None, 28*28])
y = tf.placeholder(tf.float32, [None, 10])
train_mode = tf.placeholder(tf.bool, name="train_mode")

bn_params = {
    'is_training': train_mode,
    'decay': 0.9,
    'updates_collections': None
}

# not use arg_scope
# layer1 = fully_connected(inputs=x, num_outputs=256, activation_fn=tf.nn.relu, normalizer_fn=batch_norm,
#                          normalizer_params=bn_params, weights_initializer=tf.contrib.layers.xavier_initializer(),
#                          biases_initializer=None)
# layer1 = dropout(inputs=layer1, keep_prob=keep_prop, is_training=train_mode)
# layer2 = fully_connected(inputs=layer1, num_outputs=128, activation_fn=tf.nn.relu, normalizer_fn=batch_norm,
#                          normalizer_params=bn_params, weights_initializer=tf.contrib.layers.xavier_initializer(),
#                          biases_initializer=None)
# layer2 = dropout(inputs=layer2, keep_prop=keep_prop, is_training=train_mode)
# layer3 = fully_connected(inputs=layer2, num_outputs=64, activation_fn=tf.nn.relu, normalizer_fn=batch_norm,
#                          normalizer_params=bn_params, weights_initializer=tf.contrib.layers.xavier_initializer(),
#                          biases_initializer=None)
# layer3 = dropout(inputs=layer3, keep_prop=keep_prop, is_training=train_mode)
# model_output = fully_connected(inputs=layer3, num_outputs=10, activation_fn=tf.nn.relu, normalizer_fn=batch_norm,
#                                normalizer_params=bn_params, weights_initializer=tf.contrib.layers.xavier_initializer(),
#                                biases_initializer=None)

# use arg_scope
with arg_scope([fully_connected], activation_fn=tf.nn.relu, normalizer_fn=batch_norm, normalizer_params=bn_params,
               weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer=None):
    layer1 = fully_connected(x, num_outputs=256, scope="h1")
    layer1 = dropout(layer1, keep_prop, is_training=train_mode)
    layer2 = fully_connected(layer1, num_outputs=128, scope="h2")
    layer2 = dropout(layer2, keep_prop, is_training=train_mode)
    layer3 = fully_connected(layer2, num_outputs=64, scope="h3")
    layer3 = dropout(layer3, keep_prop, is_training=train_mode)
    model_output = fully_connected(layer3, num_outputs=10, scope="model_output")

#set loss and train_step
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_output, labels=y))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(epoches):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        feed_dict_train = {x: batch_x, y: batch_y, train_mode: True}
        feed_dict_cost = {x: batch_x, y: batch_y, train_mode: False}
        optimizer = sess.run(train_step, feed_dict=feed_dict_train)
        tmp_loss = sess.run(loss, feed_dict=feed_dict_cost)
        avg_cost += tmp_loss / total_batch
    print("Epoch:", "%4d" %(epoch + 1), "cost = ", "{:.9f}".format(avg_cost))
print("learning finished!!!")

# test model
correct_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(model_output, 1)), tf.float32))
print("Accuracy:", sess.run(correct_acc, feed_dict={x: mnist.test.images, y: mnist.test.labels, train_mode: False}))
