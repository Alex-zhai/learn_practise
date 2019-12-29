import numpy as np
import gym

from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.models import Model
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

ENV_NAME = 'Pendulum-v0'
gym.undo_logger_setup()
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
action_nums = env.action_space.shape[0]
state_dims = env.observation_space.shape   # (3,)


def get_actor_model():
    input = Input(shape=(1,) + state_dims)
    x = Flatten()(input)
    x = Dense(16)(x)
    x = Activation('relu')(x)
    x = Dense(16)(x)
    x = Activation('relu')(x)
    x = Dense(16)(x)
    x = Activation('relu')(x)
    x = Dense(action_nums)(x)
    output = Activation('linear')(x)
    model = Model(inputs=input, outputs=output)
    return model


def get_critic_model():
    action_input = Input(shape=(action_nums,), name='action_input')
    state_input = Input(shape=(1, ) + state_dims, name='state_input')
    flatten_state = Flatten()(state_input)
    x = Concatenate()([action_input, flatten_state])
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    output = Activation('linear')(x)
    model = Model(inputs=[action_input, state_input], outputs=output)
    return action_input, model

actor_model = get_actor_model()
action_input, critic_model = get_critic_model()

memory = SequentialMemory(limit=100000, window_length=1)
process = OrnsteinUhlenbeckProcess(theta=0.1, mu=0., sigma=.3, size=action_nums)
ddpg_agent = DDPGAgent(nb_actions=action_nums, actor=actor_model, critic=critic_model,
                       critic_action_input=action_input, memory=memory, gamma=.99,
                       nb_steps_warmup_actor=100, nb_steps_warmup_critic=100, random_process=process,
                       target_model_update=.0001)
ddpg_agent.compile(optimizer=Adam(lr=.001, clipnorm=1.), metrics=['mae'])
ddpg_agent.fit(env, nb_steps=50000, verbose=1, nb_max_episode_steps=200)
ddpg_agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
ddpg_agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)