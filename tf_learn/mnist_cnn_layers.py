import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

# set hyper parameters
learning_rate = 0.001
epoch_size = 15
batch_size = 128

# set model class
class cnn_model_layer:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._create_model()

    def _create_model(self):
        with tf.variable_scope(self.name):
            self.istrain = tf.placeholder(tf.bool)
            self.x = tf.placeholder(tf.float32, [None, 28*28])
            self.y = tf.placeholder(tf.float32, [None, 10])
            reshape_x = tf.reshape(self.x, [-1, 28, 28, 1])
            conv1 = tf.layers.conv2d(inputs=reshape_x, filters=32, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, padding="SAME")
            drop1 = tf.layers.dropout(pool1, rate=0.7, training=self.istrain)
            conv2 = tf.layers.conv2d(inputs=drop1, filters=64, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding="SAME")
            drop2 = tf.layers.dropout(inputs=pool2, rate=0.7, training=self.istrain)
            conv3 = tf.layers.conv2d(inputs=drop2, filters=128, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2, padding="SAME")
            drop3 = tf.layers.dropout(pool3, rate=0.7, training=self.istrain)
            drop3_flat = tf.reshape(drop3, [-1, 4*4*128])
            flat1 = tf.layers.dense(inputs=drop3_flat, units=625, activation=tf.nn.relu)
            flat1_drop = tf.layers.dropout(inputs=flat1, rate=0.5, training=self.istrain)
            self.model_logit = tf.layers.dense(inputs=flat1_drop, units=10)

        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.model_logit))
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y, 1), tf.argmax(self.model_logit,1)),
                                               tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.model_logit, feed_dict={self.x: x_test, self.istrain: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy, feed_dict={self.x: x_test, self.y: y_test, self.istrain: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.train_step, self.cost], feed_dict={self.x: x_data, self.y:y_data,
                                                                      self.istrain: training})

sess = tf.Session()
model = cnn_model_layer(sess, "cnn_model_layer")
sess.run(tf.global_variables_initializer())

for epoch in range(epoch_size):
    avg_cost = 0.0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, tmp_loss = model.train(batch_x, batch_y)
        avg_cost += tmp_loss / total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print("Accuracy is:", model.get_accuracy(mnist.test.images,mnist.test.labels))