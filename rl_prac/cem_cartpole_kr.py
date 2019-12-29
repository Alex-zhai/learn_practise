import numpy as np
import gym
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.models import Sequential

from rl.agents.cem import CEMAgent
from rl.memory import EpisodeParameterMemory

ENV_NAME = 'CartPole-v0'

env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)

action_nums = env.action_space.n
state_dims = env.observation_space.shape[0]
print(action_nums, state_dims)

# define model
model = Sequential()
model.add(Flatten(input_shape=(1, ) + env.observation_space.shape))  # input_shape [1,4]
model.add(Dense(action_nums))
model.add(Activation('softmax'))

memory = EpisodeParameterMemory(limit=1000, window_length=1)
cem_agent = CEMAgent(model=model, nb_actions=action_nums, memory=memory, batch_size=50, nb_steps_warmup=2000,
                     train_interval=50, elite_frac=0.05)
cem_agent.compile()
cem_agent.fit(env, nb_steps=100000, visualize=False, verbose=2)
cem_agent.save_weights('cem_{}_params.h5f'.format(ENV_NAME), overwrite=True)
cem_agent.test(env, nb_episodes=5, visualize=True)