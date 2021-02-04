import gym
from torch import optim

import models
import actor_critic
import train
import policy

env = gym.make('CartPole-v1')
nn = models.MLPPolicyAV(env)
optimizer = optim.Adam(nn.parameters(), lr=0.01)
updater = actor_critic.ACUpdater(optimizer, gamma=.99)
policy = policy.NNPolicyAV(nn, updater)

train.train_loop(env=env, policy=policy, n_episodes=1000, render=True)
