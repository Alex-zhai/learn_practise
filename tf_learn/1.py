import tensorflow as tf
import numpy as np

a = np.random.normal(0, 0.1, (3, 4, 5))
print(a[:, -1, :].shape)

# sess = tf.Session()
#
# b = sess.run(tf.unstack(a))
# print(b)