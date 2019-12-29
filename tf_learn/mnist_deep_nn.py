import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 128
learning_rate = 0.001
epoches = 50

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

# set placeholder
x = tf.placeholder(tf.float32, [None, 28*28])
y = tf.placeholder(tf.float32, [None, 10])

# set weight and bias

w1 = tf.get_variable("w1", shape=[28*28, 512], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]))

w2 = tf.get_variable("w2", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))

w3 = tf.get_variable("w3", shape=[512, 256], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([256]))

w4 = tf.get_variable("w4", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([256]))

w5 = tf.get_variable("w5", shape=[256, 128], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([128]))

w6 = tf.get_variable("w6", shape=[128, 128], initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([128]))

w7 = tf.get_variable("w7", shape=[128, 10], initializer=tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.random_normal([10]))

def model(input_x):
    layer1 = tf.nn.relu(tf.matmul(input_x, w1) + b1)
    layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
    layer3 = tf.nn.relu(tf.matmul(layer2, w3) + b3)
    layer4 = tf.nn.relu(tf.matmul(layer3, w4) + b4)
    layer5 = tf.nn.relu(tf.matmul(layer4, w5) + b5)
    layer6 = tf.nn.relu(tf.matmul(layer5, w6) + b6)
    model_output = tf.matmul(layer6, w7) + b7
    return model_output

logits = model(x)

#set loss and train_step
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(epoches):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, tmp_loss = sess.run([train_step, loss], feed_dict={x: batch_x, y:batch_y})
        avg_cost += tmp_loss / total_batch
    print("Epoch:", "%4d" %(epoch + 1), "cost = ", "{:.9f}".format(avg_cost))
print("learning finished!!!")

# test model
correct_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1)), tf.float32))
print("Accuracy:", sess.run(correct_acc, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

r_index = random.randint(0, mnist.test.num_examples - 1)
print("true label:", sess.run(tf.argmax(mnist.test.labels[r_index, r_index+1], 1)))
print("pred label:", sess.run(tf.argmax(logits, 1), feed_dict={x: mnist.test.images[r_index:r_index+1],
                                                                    y: mnist.test.labels[r_index:r_index+1]}))