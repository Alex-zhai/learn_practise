from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection

X_FEATURE = 'x'


def my_model(features, labels, mode):
    net = features[X_FEATURE]
    for units in [10, 20, 10]:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    logits = tf.layers.dense(net, 3, activation=None)

    predicted_class = tf.argmax(logits, axis=1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class': predicted_class,
            'prob': tf.nn.softmax(logits)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    one_hot_labels = tf.one_hot(labels, 3, 1, 0)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.exponential_decay(
            learning_rate=0.1, global_step=global_step,
            decay_steps=100, decay_rate=0.001
        )
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss,train_op=train_op)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predicted_class
        )
    }
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    iris = datasets.load_iris()
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    classifier = tf.estimator.Estimator(model_fn=my_model)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={X_FEATURE: x_train}, y=y_train, num_epochs=None, shuffle=True
    )
    classifier.train(input_fn=train_input_fn, steps=1000)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={X_FEATURE: x_test}, y=y_test, num_epochs=1, shuffle=False
    )
    predictions = classifier.predict(input_fn=test_input_fn)
    y_preds = np.array(list(p['class'] for p in predictions))
    y_preds = y_preds.reshape(np.array(y_test).shape)

    score = metrics.accuracy_score(y_true=y_test, y_pred=y_preds)
    print('Accuracy (sklearn): {0:f}'.format(score))

    score = classifier.evaluate(input_fn=test_input_fn)
    print('Accuracy (tf): {0:f}'.format(score['accuracy']))

if __name__ == '__main__':
    tf.app.run()