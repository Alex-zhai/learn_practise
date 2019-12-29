from __future__ import division, print_function, absolute_import

import time

import tensorflow as tf
from tensorflow.contrib import rnn
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

start_time = time.time()

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
num_gpus = 4
num_steps = 10000
learning_rate = 0.001
batch_size = 128
display_step = 200

# Network Parameters
num_input = 28  # MNIST data input (img shape: 28*28)
timesteps = 28  # timesteps
num_hidden = 128  # hidden layer num of features
num_classes = 10  # MNIST total classes (0-9 digits)

weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def Build_rnn_cell():
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell = rnn.DropoutWrapper(lstm_cell, input_keep_prob=1.0, output_keep_prob=0.75)
    return lstm_cell


# Build a rnn network
def rnn_net(x, reuse):
    # Define a scope for reusing the variables
    with tf.variable_scope('RnnNet', reuse=reuse):
        mlstem_cell = rnn.MultiRNNCell([Build_rnn_cell() for _ in range(2)], state_is_tuple=True)
        initial_state = mlstem_cell.zero_state(batch_size, dtype=tf.float32)
        outputs, _ = tf.nn.dynamic_rnn(mlstem_cell, inputs=x, initial_state=initial_state, time_major=False)
        outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
        return tf.matmul(outputs[-1], weights['out']) + biases['out']


# Build a rnn network
# def rnn_net(x, reuse):
#     # Define a scope for reusing the variables
#     with tf.variable_scope('RnnNet', reuse=reuse):
#         # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
#         # Reshape to match picture format [Height x Width x Channel]
#         # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
#         x = tf.unstack(x, timesteps, 1)
#
#         # Define a lstm cell with tensorflow
#         lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
#
#         # Get lstm cell output
#         outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
#
#         # Linear activation, using rnn inner loop last output
#         return tf.matmul(outputs[-1], weights['out']) + biases['out']


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


# By default, all variables will be placed on '/gpu:0'
# So we need a custom device function, to assign all variables to '/cpu:0'
# Note: If GPUs are peered, '/gpu:0' can be a faster option
PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']


def assign_to_device(device, ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device

    return _assign


# Place all ops on CPU by default
with tf.device('/cpu:0'):
    tower_grads = []
    reuse_vars = False

    # tf Graph input
    X = tf.placeholder(tf.float32, [None, timesteps, num_input])
    Y = tf.placeholder(tf.float32, [None, num_classes])

    # Loop over all GPUs and construct their own computation graph
    for i in range(num_gpus):
        with tf.device(assign_to_device('/gpu:{}'.format(i), ps_device='/cpu:0')):

            # Split data between GPUs
            _x = X[i * batch_size: (i + 1) * batch_size]
            _y = Y[i * batch_size: (i + 1) * batch_size]

            # Because Dropout have different behavior at training and prediction time, we
            # need to create 2 distinct computation graphs that share the same weights.

            # Create a graph for training
            logits_train = rnn_net(_x, reuse=reuse_vars)
            # Create another graph for testing that reuse the same weights
            logits_test = rnn_net(_x, reuse=True)

            # Define loss and optimizer (with train logits, for dropout to take effect)
            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=logits_train, labels=_y))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            grads = optimizer.compute_gradients(loss_op)

            # Only first GPU compute accuracy
            if i == 0:
                # Evaluate model (with test logits, for dropout to be disabled)
                correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.argmax(_y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            reuse_vars = True
            tower_grads.append(grads)

    tower_grads = average_gradients(tower_grads)
    # noinspection PyUnboundLocalVariable
    train_op = optimizer.apply_gradients(tower_grads)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start Training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        # Keep training until reach max iterations
        for step in range(1, num_steps + 1):
            # Get a batch for each GPU
            batch_x, batch_y = mnist.train.next_batch(batch_size * num_gpus)
            batch_x = batch_x.reshape((batch_size * num_gpus, timesteps, num_input))
            # Run optimization op (backprop)
            ts = time.time()
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            te = time.time() - ts
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                # noinspection PyUnboundLocalVariable
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("Step " + str(step) + ": Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc) + ", %i Examples/sec" % int(len(batch_x) / te))
            step += 1
        print("Optimization Finished!")

        # Calculate accuracy for MNIST test images
        test_len = 128
        test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
        test_label = mnist.test.labels[:test_len]
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
end_time = time.time()
print("training time is %d" % (end_time - start_time))
