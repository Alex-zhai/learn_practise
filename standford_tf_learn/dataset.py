import tensorflow as tf
import numpy as np
from tensorflow.contrib.data import Dataset

# prepare data
x = np.random.uniform(-1, 1, (1000, 1))
y = np.power(x, 2) + np.random.normal(0, 0.1, size=x.shape)
x_train, x_test = np.split(x, [800])
y_train, y_test = np.split(y, [800])

print(x_train.shape)  # (800,1)
print(y_train.shape)

tf_x = tf.placeholder(x_train.dtype, x_train.shape)
tf_y = tf.placeholder(y_train.dtype, y_train.shape)

# create dataloader
dataset = Dataset.from_tensor_slices((tf_x, tf_y))
dataset = dataset.shuffle(buffer_size=1000)
# dataset = dataset.batch(32)
# dataset = dataset.repeat(3)
iterator = dataset.make_initializable_iterator()

bx, by = iterator.get_next()
l1 = tf.layers.dense(bx, 10, tf.nn.relu)
model_out = tf.layers.dense(l1, y.shape[1])
loss = tf.losses.mean_squared_error(labels=by, predictions=model_out)
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

sess = tf.Session()
sess.run([iterator.initializer, tf.global_variables_initializer()], feed_dict={tf_x: x_train, tf_y: y_train})

for step in range(201):
    try:
        _, train_loss = sess.run([train_step, loss])
        if step % 10 == 0:
            test_loss = sess.run(loss, feed_dict={bx: x_test, by: y_test})
            print('step: %i/200' % step, '|train loss:', train_loss, '|test loss:', test_loss)
    except tf.errors.OutOfRangeError:
        print('Finish the last epoch.')
        break