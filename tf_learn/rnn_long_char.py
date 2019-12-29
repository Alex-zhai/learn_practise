import tensorflow as tf
import numpy as np


sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

char_set = list(set(sentence))
char2dict = {w: i for i, w in enumerate(char_set)}

# set hyper_parameters
input_dim = len(char_set)
num_classes = len(char_set)
hidden_size = len(char_set)
seq_length = 10
learning_rate = 0.1

# create train x_data, y_data
x_data = []
y_data = []

for i in range(len(sentence) - seq_length):
    x_str = sentence[i:i+seq_length]
    y_str = sentence[i+1:i+seq_length+1]
    print(i, x_str, '->', y_str)
    x = [char2dict[c] for c in x_str]
    y = [char2dict[c] for c in y_str]
    x_data.append(x)
    y_data.append(y)

batch_size = len(x_data)

# set placeholder
x = tf.placeholder(tf.int32, [None, seq_length])
y = tf.placeholder(tf.int32, [None, seq_length])

x_one_hot = tf.one_hot(x, num_classes) # (none, seq_length) => (none, seq_length, num_classes)

def lstm_cell():
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
    return cell

multi_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)
output, _state = tf.nn.dynamic_rnn(multi_cells, x_one_hot, dtype=tf.float32)

output_flat = tf.reshape(output, [-1, hidden_size])
w_flat = tf.get_variable("w_flat", [hidden_size, num_classes])
b_flat = tf.get_variable("b_flat", [num_classes])

model_logit = tf.matmul(output_flat, w_flat) + b_flat
model_logit = tf.reshape(model_logit, [batch_size, seq_length, num_classes])

weights = tf.ones([batch_size, seq_length])
seq_loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits=model_logit, targets=y, weights=weights))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(seq_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(500):
    _, tmp_loss, results = sess.run([train_step, seq_loss, model_logit], feed_dict={x: x_data, y: y_data})
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        print("step: %d" % (i+1), j, "pred str is %s:" % (''.join(char_set[t] for t in index)),
              "loss={:.9f}".format(tmp_loss))

results = sess.run(model_logit, feed_dict={x: x_data})
for j, result in enumerate(results):
    index = np.argmax(result, axis=1)
    if j == 0:
        print(''.join([char_set[t] for t in index]), end='')
    else:
        print(char_set[index[-1]], end='')
