from __future__ import print_function, division, absolute_import

import tensorflow as tf
import os.path
from tensorflow.contrib.slim.python.slim.nets import inception
from tensorflow.python.ops import array_ops
from tensorflow.python.training import saver as tf_saver

slim = tf.contrib.slim

tf.app.flags.DEFINE_string("checkpoint", "2016_08/model.ckpt", "where the pretrained model stored")
tf.app.flags.DEFINE_integer("image_size", 299, "Image size to run inference on")
tf.app.flags.DEFINE_integer("num_classes", 6012, "Number of output classes")
tf.app.flags.DEFINE_string("image_path", "test_set/0a9ed4def08fe6d1", "where the test image stored")

FLAGS = tf.app.flags.FLAGS


def process_image(image):
    image = tf.cast(tf.image.decode_jpeg(image, channels=3), tf.float32)
    image = tf.image.central_crop(image, central_fraction=0.875)
    image = tf.expand_dims(image, axis=[0])
    image = tf.image.resize_bilinear(image, [FLAGS.image_size, FLAGS.image_size], align_corners=False)
    image = tf.multiply(image, 1.0 / 127.5)
    return tf.subtract(image, 1.0)


def main(args):
    if not os.path.exists(FLAGS.checkpoint):
        tf.logging.fatal('Checkpoint %s does not exist. Have you download it?' % FLAGS.checkpoint)
    graph = tf.Graph()
    with graph.as_default():
        input_image_name = tf.placeholder(tf.string)
        processed_image = process_image(input_image_name)
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            # end_points: a dictionary from components of the network to the corresponding activation.
            logits, endpoints = inception.inception_v3(processed_image, num_classes=FLAGS.num_classes,
                                                       is_training=False)
            predictions = endpoints['multi_predictions'] = tf.nn.sigmoid(logits, name='multi_predictions')
            sess = tf.Session()
            saver = tf_saver.Saver()
            saver.restore(sess, FLAGS.checkpoint)

            logits_2 = tf.layers.conv2d(endpoints['PreLogits'], FLAGS.num_classes, [1, 1])
            logits_2 = array_ops.squeeze(logits_2, [1, 2])
            predictions_2 = endpoints['multi_predictions_2'] = tf.nn.sigmoid(logits_2, name='multi_predictions_2')
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, FLAGS.checkpoint)


if __name__ == 'main':
    tf.app.run()
