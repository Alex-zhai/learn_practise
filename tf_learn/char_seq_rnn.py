# coding life is start now
import tensorflow as tf
import numpy as np

sample = "if you want you"
idx2char = list(set(sample))  # ['t', 'f', ' ', 'y', 'w', 'n', 'i', 'u', 'o', 'a']
char2idx = {c: i for i, c in enumerate(idx2char)}

# set hyper_parameters
dict_size = len(char2idx)
hidden_size = len(char2idx)
num_classes = len(char2idx)
batch_size = 1
seq_length = len(sample) - 1  # 14
learning_rate = 0.1

sample_idx = [char2idx[c] for c in sample]  #[6, 1, 2, 3, 8, 7, 2, 4, 9, 5, 0, 2, 3, 8, 7]
x_data = [sample_idx[:-1]]
y_data = [sample_idx[1:]]

# set placeholder
x = tf.placeholder(tf.int32, [None, seq_length])
y = tf.placeholder(tf.int32, [None, seq_length])

x_one_hot = tf.one_hot(x, num_classes)

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
output, _states = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=initial_state, dtype=tf.float32)

output_flat = tf.reshape(output, [-1, hidden_size])
w_flat = tf.get_variable("w_flat", [hidden_size, num_classes])
b_flat = tf.get_variable("b_flat", [num_classes])
model_logit = tf.matmul(output_flat, w_flat) + b_flat
model_logit = tf.reshape(model_logit, [batch_size, seq_length, num_classes])

# set loss function
weights = tf.ones([batch_size, seq_length])
seq_loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits=model_logit, targets=y, weights=weights))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(seq_loss)

prediction = tf.argmax(model_logit, axis=2)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(50):
    _, tmp_loss = sess.run([train_step, seq_loss], feed_dict={x: x_data, y: y_data})
    result = sess.run(prediction, feed_dict={x: x_data})

    result_str = [idx2char[idx] for idx in np.squeeze(result)]
    print("step: %d" % (i+1), "loss={:.9f}".format(tmp_loss), "pred str is: %s" % (''.join(result_str)))