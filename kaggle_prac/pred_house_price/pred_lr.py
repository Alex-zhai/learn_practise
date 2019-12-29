# coding:utf-8
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from scipy.stats import skew



# def scatter(data, p_name):
#     matplotlib.rcParams['figure.figsize'] = (12.0, 4.0)
#     data = pd.concat([data['SalePrice'], data[p_name]], axis=1)
#     data.plot.scatter(x=p_name, y='SalePrice', ylim=(0, 800000))
#
# data = pd.read_csv("train.csv")
# matplotlib.rcParams['figure.figsize'] = (16.0, 6.0)
# prices = pd.DataFrame({"log(price+1)": np.log1p(data["SalePrice"]), "price": data["SalePrice"]})
# prices.hist()
#
# matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
# corrmat = data.corr()
# sns.heatmap(corrmat, vmax=.8, square=True)
#
# scatter(data, 'OverallQual')

def load_file(is_test=False):
    test = pd.read_csv("test.csv")
    train = pd.read_csv("train.csv")
    cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    data_df = pd.concat((train.loc[:, cols], test.loc[:, cols]))
    numeric_feats = data_df.dtypes[data_df.dtypes != 'object'].index  # 取的是num类型属性名称
    skewed_feats = data_df[numeric_feats].apply(lambda x: skew(x.dropna()))
    #print(skewed_feats)
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    print(skewed_feats)
    skewed_feats = skewed_feats.index
    #print(skewed_feats)
    data_df[skewed_feats] = np.log1p(data_df[skewed_feats])
    print(data_df.head())
    data_df = pd.get_dummies(data_df)
    data_df = data_df.fillna(data_df.mean())

    return data_df, train, test

all_data, train, test = load_file()
x_train = all_data[:train.shape[0]]
x_test = all_data[train.shape[0]:]
y_train = np.expand_dims(np.log1p(train.SalePrice), axis=1)

print(x_train.shape, y_train.shape)
print(all_data.head())

learning_rate = 0.001
n_input = x_train.shape[1]
epoch_nums = 2000
step_size = 50

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, 1])

w = tf.Variable(tf.random_normal([n_input, 1], stddev=0.01))
b = tf.Variable(tf.random_normal([1], stddev=0.01))

model_logit = tf.matmul(x, w) + b

loss = tf.reduce_mean(tf.square(model_logit - y))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(epoch_nums):
    avg_cost = 0.0
    for step in range(0, len(x_train), step_size):
        _, temp_loss = sess.run([train_step, loss], feed_dict={x: x_train[step: step + step_size],
                                                               y: y_train[step: step + step_size]})
        avg_cost += temp_loss
    avg_cost = avg_cost / len(x_train)
    if epoch % 250 ==0:
        print("epoch: {} cost: {:.4f}".format(epoch, avg_cost))
print("Optimization Finished!")

test_price = np.squeeze(np.expm1(sess.run(model_logit, feed_dict={x: x_test})), axis=(1,))
train_price = np.squeeze(np.expm1(sess.run(model_logit, feed_dict={x: x_train})), axis=(1,))

ids = pd.read_csv("test.csv").Id

submission = ['Id,SalePrice']

for id, prediction in zip(ids, test_price):
    submission.append('{0},{1}'.format(id, prediction))

submission = '\n'.join(submission)

with open('submission.csv', 'w') as outfile:
    outfile.write(submission)