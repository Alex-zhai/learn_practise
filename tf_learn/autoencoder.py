import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# set parameters
batch_size = 128
learning_rate = 0.01
num_steps = 10000
display_freq = 100
image_num_to_show = 10

input_size = 28 * 28
encoder_layer1_num = 256
encoder_layer2_num = 128

x = tf.placeholder("float", shape=[None, 784])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([input_size, encoder_layer1_num])),
    'encoder_h2': tf.Variable(tf.random_normal([encoder_layer1_num, encoder_layer2_num])),
    'decoder_h1': tf.Variable(tf.random_normal([encoder_layer2_num, encoder_layer1_num])),
    'decoder_h2': tf.Variable(tf.random_normal([encoder_layer1_num, input_size])),
}

bias = {
    'encoder_b1': tf.Variable(tf.random_normal([encoder_layer1_num])),
    'encoder_b2': tf.Variable(tf.random_normal([encoder_layer2_num])),
    'decoder_b1': tf.Variable(tf.random_normal([encoder_layer1_num])),
    'decoder_b2': tf.Variable(tf.random_normal([input_size])),
}

def encoder(input_x):
    en_layer1 = tf.nn.sigmoid(tf.add(tf.matmul(input_x, weights['encoder_h1']), bias['encoder_b1']))
    en_layer2 = tf.nn.sigmoid(tf.add(tf.matmul(en_layer1, weights['encoder_h2']), bias['encoder_b2']))
    return en_layer2

def decoder(input_x):
    de_layer1 = tf.nn.sigmoid(tf.add(tf.matmul(input_x, weights['decoder_h1']), bias['decoder_b1']))
    de_layer2 = tf.nn.sigmoid(tf.add(tf.matmul(de_layer1, weights['decoder_h2']), bias['decoder_b2']))
    return de_layer2

encoder_output = encoder(x)
decoder_output = decoder(encoder_output)

# define loss function
loss = tf.reduce_mean(tf.pow(x - decoder_output, 2))
train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

for step in range(num_steps):
    batch_x, _ = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={x: batch_x})
    l = sess.run(loss, feed_dict={x: batch_x})
    if (step+1) % display_freq == 0:
        print('Step %i: Minibatch loss: %f' %(step+1, l))


n = 4
canvas_orig = np.empty((28 * n, 28 * n))
canvas_recon = np.empty((28 * n, 28 * n))
for i in range(n):
    # MNIST test set
    batch_x, _ = mnist.test.next_batch(n)
    # Encode and decode the digit image
    g = sess.run(decoder_output, feed_dict={x: batch_x})

    # Display original images
    for j in range(n):
        # Draw the original digits
        canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
        batch_x[j].reshape([28, 28])
        # Display reconstructed images
    for j in range(n):
        # Draw the reconstructed digits
        canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
        g[j].reshape([28, 28])

print("Original Images")
plt.figure(figsize=(n, n))
plt.imshow(canvas_orig, origin="upper", cmap="gray")
plt.show()

print("Reconstructed Images")
plt.figure(figsize=(n, n))
plt.imshow(canvas_recon, origin="upper", cmap="gray")
plt.show()