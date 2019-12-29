import tensorflow as tf
import numpy as np
import pandas as pd

def MinMaxScale(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

def load_file(is_test):
    if is_test:
        data_df = pd.read_csv("test.csv")
    else:
        data_df = pd.read_csv("train.csv")

    cols = ["Pclass", "Sex", "Age", "Fare", "Embarked_0", "Embarked_1", "Embarked_2"]
    data_df["Sex"] = data_df["Sex"].map({'female': 0, 'male': 1}).astype(int)

    # handle missing values of 'Age'
    data_df["Age"] = data_df["Age"].fillna(data_df["Age"].mean())
    data_df["Fare"] = data_df["Fare"].fillna(data_df["Fare"].mean())

    data_df["Embarked"] = data_df["Embarked"].fillna('S')
    data_df["Embarked"] = data_df["Embarked"].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    data_df = pd.concat([data_df, pd.get_dummies(data_df["Embarked"], prefix="Embarked")], axis=1)
    data = data_df[cols].values
    if is_test:
        sing_col = data_df["PassengerId"].values
    else:
        sing_col = data_df["Survived"].values
    return sing_col, data

y_train, x_train = load_file(0)
y_train = np.expand_dims(y_train, 1)
train_len = len(x_train)
pred_pass_id, x_test = load_file(1)
x_all_data = np.vstack((x_train, x_test))
scale_data = MinMaxScale(x_all_data)
x_train = scale_data[:train_len]
x_test = scale_data[train_len:]
print(x_train.shape, x_test.shape)

# set hyper_parameters
learning_rate = 0.1
input_nums = 7
# n_hidden1 = 256
# n_hidden2 = 128
# n_hidden3 = 64
train_epoches = 20
step_size = 1000
batch_size = 32
display_freq = 1

# set placeholder
x = tf.placeholder(tf.float32, [None, input_nums])
y = tf.placeholder(tf.float32, [None, 1])

# set weights and bias

# w1 = tf.Variable(tf.random_normal([input_nums, n_hidden1], stddev=0.01))
# b1 = tf.Variable(tf.random_normal([n_hidden1], stddev=0.01))
# w2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2], stddev=0.01))
# b2 = tf.Variable(tf.random_normal([n_hidden2], stddev=0.01))
# w3 = tf.Variable(tf.random_normal([n_hidden2, n_hidden3], stddev=0.01))
# b3 = tf.Variable(tf.random_normal([n_hidden3], stddev=0.01))
w4 = tf.Variable(tf.random_normal([input_nums, 1], stddev=0.01))
b4 = tf.Variable(tf.random_normal([1], stddev=0.01))

# create mlp_model
def mlp_model(input_x):
    # layer1 = tf.nn.relu(tf.matmul(input_x, w1) + b1)
    # layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
    # layer3 = tf.nn.relu(tf.matmul(layer2, w3) + b3)
    model_logit = tf.matmul(input_x, w4) + b4
    return model_logit

model_logit = mlp_model(x)
model_pred = tf.sigmoid(model_logit)
# set loss and train_step
loss = -tf.reduce_mean(y*tf.log(model_pred) + (1-y)*tf.log(1-model_pred))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# set pred operation
pred_value = tf.cast(model_pred > 0.5, dtype=tf.float32)
pred_acc = tf.reduce_mean(tf.cast(tf.equal(y, pred_value), dtype=tf.float32))

#start training!!!
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(train_epoches):
    avg_cost = 0.0
    avg_acc = 0.0
    for step in range(step_size):
        off_set = (step * batch_size) % (y_train.shape[0] - batch_size)
        batch_x = x_train[off_set: (off_set + batch_size), :]
        batch_y = y_train[off_set: (off_set + batch_size), :]
        _, tmp_loss, acc = sess.run([train_step, loss, pred_acc], feed_dict={x: batch_x, y: batch_y})
        avg_cost += tmp_loss / step_size
        avg_acc += acc / step_size
    if epoch % display_freq == 0:
        print("Epoch %d" % (epoch + 1), "cost = {:.4f}".format(avg_cost), "train accuracy = {:.4f}".format(avg_acc))
print("Training Finished!!!")

outputs = sess.run(pred_value, feed_dict={x: x_test})
submission = ['PassengerId,Survived']
for id, prediction in zip(pred_pass_id, outputs):
    submission.append('{0},{1}'.format(id, int(prediction)))
submission = '\n'.join(submission)
with open('submission.csv', 'w') as outfile:
    outfile.write(submission)


