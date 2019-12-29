import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

learning_rate = 0.001
epoches = 20
batch_size = 128

# set placeholder
x = tf.placeholder(tf.float32, [None, 28*28])
y = tf.placeholder(tf.float32, [None, 10])

# set weight  cnn: weight is kernel
w_kernel1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
w_kernel2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
w_flat = tf.Variable(tf.random_normal([7*7*64, 10]))
b_flat = tf.Variable(tf.random_normal([10]))

def model(input_x):
    input_x = tf.reshape(input_x, [-1, 28, 28, 1])
    layer1 = tf.nn.conv2d(input_x, w_kernel1, strides=[1, 1, 1, 1], padding='SAME')
    layer1 = tf.nn.relu(layer1)
    layer1 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    layer2 = tf.nn.conv2d(layer1, w_kernel2, strides=[1, 1, 1, 1], padding='SAME')
    layer2 = tf.nn.relu(layer2)
    layer2 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    layer2_flat = tf.reshape(layer2, [-1, 7*7*64])
    model_output = tf.matmul(layer2_flat, w_flat) + b_flat
    return model_output

model_logits = model(x)

# define loss and train step
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_logits, labels=y))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# start training
for epoch in range(epoches):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, tmp_loss = sess.run([train_step, loss], feed_dict={x: batch_x, y: batch_y})
        avg_cost += tmp_loss / total_batch
    print("Epoch:", "%04d" % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

print("learning finished!!!")

correct_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(model_logits, 1), tf.argmax(y, 1)), tf.float32))
print("Accuracy is:", sess.run(correct_acc, feed_dict={x: mnist.test.images, y: mnist.test.labels}))