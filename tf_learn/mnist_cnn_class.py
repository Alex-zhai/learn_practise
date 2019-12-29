import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

# set hyper parameters
learning_rate = 0.001
epoches = 20
batch_size = 128

class CNNModel:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.x = tf.placeholder(tf.float32, [None, 28*28])
            self.y = tf.placeholder(tf.float32, [None, 10])
            self.keep_prob = tf.placeholder(tf.float32)
            resize_x = tf.reshape(self.x, [-1, 28, 28, 1])
            w1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
            layer1 = tf.nn.conv2d(resize_x, w1, strides=[1, 1, 1, 1], padding="SAME")
            layer1 = tf.nn.relu(layer1)
            layer1 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            layer1 = tf.nn.dropout(layer1, keep_prob=self.keep_prob)
            w2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
            layer2 = tf.nn.conv2d(layer1, w2, strides=[1, 1, 1, 1], padding="SAME")
            layer2 = tf.nn.relu(layer2)
            layer2 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            layer2 = tf.nn.dropout(layer2, keep_prob=self.keep_prob)
            w3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
            layer3 = tf.nn.conv2d(layer2, w3, strides=[1, 1, 1, 1], padding="SAME")
            layer3 = tf.nn.relu(layer3)
            layer3 = tf.nn.max_pool(layer3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            layer3 = tf.nn.dropout(layer3, keep_prob=self.keep_prob)
            w_flat1 = tf.get_variable("w_flat1", shape=[4*4*128, 625],
                                      initializer=tf.contrib.layers.xavier_initializer())
            b_flat1 = tf.Variable(tf.random_normal([625]))
            layer3_flat = tf.reshape(layer3, [-1, 4*4*128])
            layer4 = tf.matmul(layer3_flat, w_flat1) + b_flat1
            layer4 = tf.nn.relu(layer4)
            w_flat2 = tf.get_variable("w_flat2", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
            b_flat2 = tf.Variable(tf.random_normal([10]))
            self.model_logit = tf.matmul(layer4, w_flat2) + b_flat2
        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.model_logit, labels=self.y))
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y, 1), tf.argmax(self.model_logit, 1)),
                                               tf.float32))

    def predict(self, x_test, keep_prop=1.0):
        return self.sess.run(self.model_logit, feed_dict={self.x: x_test, self.keep_prob: keep_prop})

    def get_accuracy(self, x_test, y_test, keep_prop=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.x: x_test, self.y: y_test, self.keep_prob: keep_prop})

    def train(self, x_data, y_data, keep_prop=0.7):
        return self.sess.run([self.train_step, self.cost], feed_dict={self.x: x_data, self.y: y_data,
                                                                      self.keep_prob: keep_prop})

sess = tf.Session()
model = CNNModel(sess, "cnnmodel")
sess.run(tf.global_variables_initializer())

for epoch in range(epoches):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, tmp_loss = model.train(batch_x, batch_y)
        avg_cost += tmp_loss / total_batch
    print("epoch:", "%04d" % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
print("learning finished!!")

print("Accuracy is:", model.get_accuracy(mnist.test.images, mnist.test.labels))