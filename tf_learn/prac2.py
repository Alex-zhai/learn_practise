import tensorflow as tf
import numpy as np

def add_layer(input, in_size, out_size, activation = None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    bias = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    logits = tf.matmul(input, weights) + bias
    if activation is None:
        outputs = logits
    else:
        outputs = activation(logits)
    return outputs

x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:,np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_date = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

def model(input):
    layer1 = add_layer(input, 1, 10, activation=tf.nn.relu)
    layer2 = add_layer(layer1,10, 1, activation=None)
    return layer2

model_logit = model(xs)

loss = tf.reduce_mean(tf.square(ys - model_logit))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_date})
    if i % 50 == 0:
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_date}))