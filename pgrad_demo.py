import gym
import torch
from torch import optim

import pgrad

env = gym.make('CartPole-v1')
env.seed(1)
torch.manual_seed(1)

nn = pgrad.MLPPolicy(env)
optimizer = optim.Adam(nn.parameters(), lr=0.01)
updater = pgrad.PGUpdater(optimizer, gamma=.99)
policy = pgrad.NNPolicy(nn, updater)

pgrad.train_loop(env=env, policy=policy, n_eposodes=1000)
