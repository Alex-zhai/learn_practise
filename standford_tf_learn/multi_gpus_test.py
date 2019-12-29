from __future__ import print_function
import tensorflow as tf

with tf.device('/cpu:0'):
    a = [0, 1, 2, 3]

for d in ['/device:GPU:0', '/device:GPU:1', '/device:GPU:2', '/device:GPU:3']:
    i = 0
    with tf.device(d):
        sum_op = tf.multiply(a[i], a[i])
        i += 1

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(sum_op))