import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


print(tf.__version__)
n_data = np.ones((100, 2))
x0 = np.random.normal(2*n_data, 1)  #(100,2)
y0 = np.zeros(100)
x1 = np.random.normal(-2*n_data, 1)
y1 = np.ones(100)

x = np.vstack((x0, x1))  # (200,2)
print(x.shape)
y = np.hstack((y0, y1))  # (200,)
print(y.shape)

plt.scatter(x[:, 0], x[:, 1], c=y, s=100, lw=0, cmap='RdYlGn')
plt.show()

tf_x = tf.placeholder(tf.float32, x.shape)
tf_y = tf.placeholder(tf.int32, y.shape)

# model
l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)
model_out = tf.layers.dense(l1, 2)

# loss and train_step
loss = tf.losses.sparse_softmax_cross_entropy(labels=tf_y, logits=model_out)
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)
accuracy = tf.metrics.accuracy(labels=tf.squeeze(tf_y), predictions=tf.argmax(model_out, axis=1))[1]
print(accuracy)

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

plt.ion()

for step in range(100):
    _,temp_loss, accu, pred = sess.run([train_step, loss, accuracy, model_out], feed_dict={tf_x: x, tf_y: y})
    if step % 2 == 0:
        print("step %d loss is %f" %(step, temp_loss))
        plt.cla()
        plt.scatter(x[:, 0], x[:, 1], c=pred.argmax(1), s=100, lw=0, cmap='RdYlGn')
        plt.text(1.5, -4, 'Accuracy=%.2f' % accu, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()