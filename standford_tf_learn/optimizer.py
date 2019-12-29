import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# set hyper_parameters
LR = 0.01
BATCH_SIZE = 32

x = np.linspace(-1, 1, 100)[:, np.newaxis]  #(100,1)
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise

plt.scatter(x, y)
plt.show()

class Net:
    def __init__(self, opt, **kwargs):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        input = tf.layers.dense(inputs=self.x, units=20, activation=tf.nn.relu)
        output = tf.layers.dense(inputs=input, units=1)
        self.loss = tf.losses.mean_squared_error(labels=self.y, predictions=output)
        self.train = opt(LR, **kwargs).minimize(self.loss)

net_SGD = Net(tf.train.GradientDescentOptimizer)
net_Momentum = Net(tf.train.MomentumOptimizer, momentum=0.9)
net_RMSprop = Net(tf.train.RMSPropOptimizer)
net_Adam = Net(tf.train.AdamOptimizer)

nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

losses_his = [[], [], [], []]

for step in range(300):          # for each training step
    index = np.random.randint(0, x.shape[0], BATCH_SIZE)
    b_x = x[index]
    b_y = y[index]

    for net, l_his in zip(nets, losses_his):
        _, l = sess.run([net.train, net.loss], {net.x: b_x, net.y: b_y})
        l_his.append(l)     # loss recoder

# plot loss history
labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
for i, l_his in enumerate(losses_his):
    plt.plot(l_his, label=labels[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.show()