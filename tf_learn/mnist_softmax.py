import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data

# set parameters
learning_rate = 0.001
num_epoches = 20
batch_size = 128

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

# set placeholder
x = tf.placeholder(tf.float32, [None, 28*28])
y = tf.placeholder(tf.float32, [None, 10])

# set valiable and bias
w = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.random_normal([10]))

# create model
model_logit = tf.add(tf.matmul(x, w), b)

# set loss and train step
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=model_logit))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# set sess
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(num_epoches):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        _, tmp_loss = sess.run([train_step, loss], feed_dict={x:x_batch, y:y_batch})
        avg_cost += tmp_loss / total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))

print("learning finished!!")

# test model
correct_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(model_logit, 1)), tf.float32))
print("Accuracy:", sess.run(correct_acc, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

r_index = random.randint(0, mnist.test.num_examples - 1)
print("true label:", sess.run(tf.argmax(mnist.test.labels[r_index, r_index+1], 1)))
print("pred label:", sess.run(tf.argmax(model_logit, 1), feed_dict={x: mnist.test.images[r_index:r_index+1],
                                                                    y: mnist.test.labels[r_index:r_index+1]}))
