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
w1 = tf.Variable(tf.random_normal([28*28, 256]))
b1 = tf.Variable(tf.random_normal([256]))
w2 = tf.Variable(tf.random_normal([256, 128]))
b2 = tf.Variable(tf.random_normal([128]))
w3 = tf.Variable(tf.random_normal([128, 10]))
b3 = tf.Variable(tf.random_normal([10]))

def model(input_x):
    layer1 = tf.nn.relu(tf.matmul(input_x, w1) + b1)
    layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
    model_output = tf.matmul(layer2, w3) + b3
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