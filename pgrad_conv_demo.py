import gym
from torch import optim

import models
import pgrad
import train
import policy

env = gym.make('BreakoutDeterministic-v4')
nn = models.ConvPolicy(env)
optimizer = optim.Adam(nn.parameters(), lr=0.01)
updater = pgrad.PGUpdater(optimizer, gamma=.99)
policy = policy.NNPolicy(nn, updater)

train.train_loop(env=env, policy=policy, n_episodes=1000, render=True)
