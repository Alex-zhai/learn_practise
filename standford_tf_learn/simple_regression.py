import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# prepare data
x = np.linspace(-1, 1, 100)[:, np.newaxis]  # (100,1)
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise

# plot data
plt.scatter(x, y)
plt.show()

# prepare placeholder x and y
tf_x = tf.placeholder(tf.float32, x.shape)
tf_y = tf.placeholder(tf.float32, y.shape)

# set model
l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)
model_out = tf.layers.dense(l1, 1)

# set loss function and train step
loss = tf.losses.mean_squared_error(labels=tf_y, predictions=model_out)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train_step = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
plt.ion()

for step in range(100):
    _, temp_loss, pred = sess.run([train_step, loss, model_out], feed_dict={tf_x: x, tf_y: y})
    if step % 5 == 0:
        print("step %d loss is %f" % (step, temp_loss))
        plt.cla()
        plt.scatter(x, y)
        plt.plot(x, pred, 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % temp_loss, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()