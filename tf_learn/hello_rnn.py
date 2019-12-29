import tensorflow as tf
import numpy as np


idx2char = ['h', 'i', 'e', 'l', '0']
x_data = [[0, 1, 0, 2, 3, 3]] #hihell
x_one_hot = [[[1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0],
             [1, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 0, 1, 0]]]
y_data = [[1, 0, 2, 3, 3, 4]]

# set hyper_parameters
num_classes = 5
seq_length = 6
input_num = 5
hidden_size = 5
batch_size = 1
learning_rate = 0.1

# set placeholder
x = tf.placeholder(tf.float32, [None, seq_length, input_num])
y = tf.placeholder(tf.int32, [None, seq_length])

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
output, _state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

x_flat = tf.reshape(output, [-1, hidden_size])
w_flat = tf.get_variable("w_flat", [hidden_size, num_classes])
b_flat = tf.get_variable("b_flat", [num_classes])
model_logit = tf.matmul(x_flat, w_flat) + b_flat

model_logit = tf.reshape(model_logit, [batch_size, seq_length, num_classes])

weights = tf.ones([batch_size, seq_length])

seq_loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits=model_logit, targets=y, weights=weights))

train_step = tf.train.AdamOptimizer(learning_rate).minimize(seq_loss)

predict_seq = tf.argmax(model_logit, 2)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(50):
    _, tmp_loss = sess.run([train_step, seq_loss], feed_dict={x: x_one_hot, y: y_data})
    result_seq = sess.run(predict_seq, feed_dict={x: x_one_hot})
    print("step:", i+1, "loss=", tmp_loss, "pred seq is: ", result_seq, "true seq is:", y_data)
    result_str = [idx2char[i] for i in np.squeeze(result_seq)]
    print("\tpred str is:", ''.join(result_str))