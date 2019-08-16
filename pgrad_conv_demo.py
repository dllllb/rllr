import gym
import torch
from torch import optim

import pgrad

env = gym.make('BreakoutDeterministic-v4')
env.seed(1)
torch.manual_seed(1)

nn = pgrad.ConvPolicy(env)
optimizer = optim.Adam(nn.parameters(), lr=0.01)
updater = pgrad.PGUpdater(optimizer, gamma=.99)
policy = pgrad.NNPolicy(nn, updater)

pgrad.train_loop(env=env, policy=policy, n_episodes=1000, render=True)
