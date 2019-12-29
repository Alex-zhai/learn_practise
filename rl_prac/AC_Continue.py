import tensorflow as tf
import numpy as np
import gym

OUTPUT_GRAPH = False
MAX_EPISODE = 1000
MAX_EP_STEPS = 200
DISPLAY_REWARD_THRESHOLD = -100  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time
GAMMA = 0.9
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic


class Critic_Net(object):
    def __init__(self, sess, s_dim, lr=0.01):
        self.sess = sess
        self.s_dim = s_dim
        self.lr = lr
        with tf.variable_scope('inputs'):
            self.s = tf.placeholder(dtype=tf.float32, shape=[1, s_dim], name='s')
            self.v_next = tf.placeholder(dtype=tf.float32, shape=[1, 1], name='v_next')
            self.r = tf.placeholder(dtype=tf.float32, name='r')
        with tf.variable_scope('Critic'):
            layer1 = tf.layers.dense(inputs=self.s, units=30,
                                     activation=tf.nn.relu, kernel_initializer=tf.random_uniform_initializer(0., .1),
                                     bias_initializer=tf.constant_initializer(0.1), name='l1')
            self.v = tf.layers.dense(inputs=layer1, units=1, activation=None,
                                     kernel_initializer=tf.random_uniform_initializer(0., .1),
                                     bias_initializer=tf.constant_initializer(0.1), name='v')
            with tf.variable_scope('loss'):
                self.td_error = tf.reduce_mean(self.r + GAMMA*self.v_next - self.v)
                self.loss = tf.square(self.td_error)
            with tf.variable_scope('ANet_train'):
                self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_next = s[np.newaxis, :], s_[np.newaxis, :]
        v_next = self.sess.run(self.v, {self.s: s_next})
        td_error, _ = self.sess.run([self.td_error, self.train_op], {self.s: s, self.v_next: v_next, self.r: r})
        return td_error


class Actor_Net(object):
    def __init__(self, sess, s_dim, action_bound, lr=0.0001):
        self.sess = sess
        self.s_dim = s_dim
        self.action_bound = action_bound
        self.lr = lr

        self.s = tf.placeholder(dtype=tf.float32, shape=[1, s_dim], name='s')
        self.a = tf.placeholder(dtype=tf.float32, shape=None, name='a')
        self.td_error = tf.placeholder(dtype=tf.float32, shape=None, name='td_error')

        layer1 = tf.layers.dense(inputs=self.s, units=30, activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0., .1),
                                 bias_initializer=tf.constant_initializer(0.1),
                                 name='l1')
        mu = tf.layers.dense(inputs=layer1, units=1, activation=tf.nn.tanh,
                             kernel_initializer=tf.random_normal_initializer(0., .1),
                             bias_initializer=tf.constant_initializer(0.1),
                             name='mu')
        sigma = tf.layers.dense(inputs=layer1, units=1, activation=tf.nn.softplus,
                                kernel_initializer=tf.random_normal_initializer(0., .1),
                                bias_initializer=tf.constant_initializer(0.1),
                                name='sigma')
        global_step = tf.Variable(0, trainable=False)
        self.mu, self.sigma = tf.squeeze(mu*2), tf.squeeze(sigma + 0.1)
        self.normal_dist = tf.distributions.Normal(self.mu, self.sigma)

        self.action = tf.clip_by_value(self.normal_dist.sample(1), action_bound[0], action_bound[1])

        with tf.variable_scope('exp_v'):
            log_prob = self.normal_dist.log_prob(self.a)
            self.exp_v = log_prob * self.td_error
            self.exp_v += 0.01*self.normal_dist.entropy()

        with tf.variable_scope('train_op'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(-self.exp_v, global_step)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        _, exp_v = self.sess.run([self.train_op, self.exp_v], {self.s: s, self.a: a, self.td_error: td})
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        return self.sess.run(self.action, {self.s: s})


env = gym.make('Pendulum-v0')
env.seed(1)  # reproducible
env = env.unwrapped

N_S = env.observation_space.shape[0]
A_BOUND = env.action_space.high

sess = tf.Session()

actor = Actor_Net(sess, s_dim=N_S, action_bound=[-A_BOUND, A_BOUND], lr=LR_A)
critic = Critic_Net(sess=sess, s_dim=N_S, lr=LR_C)

sess.run(tf.global_variables_initializer())

if OUTPUT_GRAPH:
    tf.summary.FileWriter('logs/', sess.graph)

for i in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    ep_rs = []
    while True:
        env.render()
        a = actor.choose_action(s)
        s_, r, done, info = env.step(a)
        r /= 10
        td_error = critic.learn(s, r, s_)
        actor.learn(s, a, td_error)

        s = s_
        t += 1
        ep_rs.append(r)
        if t > MAX_EP_STEPS:
            ep_rs_sum = sum(ep_rs)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.9 + ep_rs_sum * 0.1
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i, "  reward:", int(running_reward))
            break
