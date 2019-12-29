import tensorflow as tf
import numpy as np

# prepare data
x_date = np.random.rand(100).astype(np.float32)
y_date = 0.1*x_date + 0.3

weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
bias = tf.Variable(tf.zeros([1]))

y = x_date*weights + bias

loss = tf.reduce_mean(tf.square(y - y_date))

optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

for i in range(1000):
    sess.run(optimizer)
    if i % 20 == 0:
        print(i, sess.run(weights), sess.run(bias))