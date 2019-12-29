import tensorflow as tf

sess = tf.Session()

dataset = tf.data.Dataset.range(10)
dataset = dataset.repeat(2).shuffle(10).batch(10)
# dataset = dataset.shuffle(10).repeat(2).batch(10)
iterator = dataset.make_one_shot_iterator()
nums = iterator.get_next()

print(sess.run(nums))
nums1 = iterator.get_next()

print(sess.run(nums1))

# dataset = dataset.repeat(2).shuffle(10).batch(10)
#
# nums2 = iterator.get_next()
#
# print(sess.run(nums2))
# nums3 = iterator.get_next()
#
# print(sess.run(nums3))