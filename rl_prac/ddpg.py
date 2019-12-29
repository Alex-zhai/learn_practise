import tensorflow as tf
import numpy as np
import gym

# set hyperParameters
MAX_EPISODE = 200
MAX_EP_STEP = 200
BATCH_SIZE = 32
ANET_LR = 0.001
CNET_LR = 0.002
GAMMA = 0.9
TAU = 0.01
MEMORY_CAPACITY = 10000

RENDER = False
ENV_NAME = 'Pendulum-v0'


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros(shape=(MEMORY_CAPACITY, 2*s_dim+a_dim+1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()
        self.replace_a_counter, self.replace_c_counter = 0, 0
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.a_bound = a_bound
        self.S = tf.placeholder(dtype=tf.float32, shape=[None, s_dim], name='s')
        self.S_Next = tf.placeholder(dtype=tf.float32, shape=[None, s_dim], name='s_next')
        self.R = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='r')

        with tf.variable_scope('Actor'):
            self.a = self._build_actor_net(self.S, scope='eval', trainable=True)
            a_next = self._build_actor_net(self.S_Next, scope='target', trainable=False)

        with tf.variable_scope('Critic'):
            q = self._build_critic_net(self.S, self.a, scope='eval', trainable=True)
            q_next = self._build_critic_net(self.S_Next, a_next, scope='target', trainable=False)

        # networks parameters
        self.cur_ANet_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.tar_ANet_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.cur_CNet_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.tar_CNet_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace_ANet = [tf.assign(tar_param, (1-TAU)*tar_param + TAU*cur_param)
                                    for cur_param, tar_param in zip(self.cur_ANet_params, self.tar_ANet_params)]
        self.soft_replace_CNet = [tf.assign(tar_param, (1-TAU)*tar_param + TAU*cur_param)
                                    for cur_param, tar_param in zip(self.cur_CNet_params, self.tar_CNet_params)]

        q_target = self.R + GAMMA*q_next

        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.CNet_trainOp = tf.train.AdamOptimizer(learning_rate=CNET_LR)\
            .minimize(td_error, var_list=self.cur_CNet_params)

        a_loss = -tf.reduce_mean(q)
        self.ANet_trainOp = tf.train.AdamOptimizer(learning_rate=ANET_LR)\
            .minimize(a_loss, var_list=self.cur_ANet_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, feed_dict={self.S: s[np.newaxis, :]})[0]

    def learn(self):
        self.sess.run(self.soft_replace_ANet)
        self.sess.run(self.soft_replace_CNet)
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        batch_trans = self.memory[indices, :]
        batch_s = batch_trans[:, :self.s_dim]
        batch_a = batch_trans[:, self.s_dim: self.s_dim + self.a_dim]
        batch_r = batch_trans[:, -self.s_dim - 1: -self.s_dim]
        batch_s_ = batch_trans[:, -self.s_dim:]
        self.sess.run(self.CNet_trainOp,
                      feed_dict={self.S: batch_s, self.a: batch_a, self.R: batch_r, self.S_Next: batch_s_})
        self.sess.run(self.ANet_trainOp, feed_dict={self.S: batch_s})

    def store_transition(self, s, a, r, s_):
        trans = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = trans
        self.pointer += 1

    def _build_actor_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(inputs=s, units=30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(inputs=net, units=self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_critic_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            layer1_units = 30
            w1_s = tf.get_variable(name='w1_s', shape=[self.s_dim, layer1_units], trainable=trainable)
            w1_a = tf.get_variable(name='w1_a', shape=[self.a_dim, layer1_units], trainable=trainable)
            b1 = tf.get_variable(name='b1', shape=[1, layer1_units], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(inputs=net, units=1, trainable=trainable)

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ddpg_model = DDPG(a_dim, s_dim, a_bound)

var = 3

for i in range(MAX_EPISODE):
    s = env.reset()
    ep_r = 0
    for j in range(MAX_EP_STEP):
        if RENDER:
            env.render()
        a = ddpg_model.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)
        s_, r, done, info = env.step(a)
        ddpg_model.store_transition(s, a, r/10, s_)
        if ddpg_model.pointer > MEMORY_CAPACITY:
            var *= .9995
            ddpg_model.learn()

        s = s_
        ep_r += r
        if j == MAX_EP_STEP-1:
            print('Episode:', i, ' Reward: %i' % int(ep_r), 'Explore: %.2f' % var, )
            if ep_r > -300:
                RENDER = True
            break