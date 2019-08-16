import gym
import torch

import models
from pgrad import PGUpdater
from train import train_loop
from policy import NNPolicy


def test_train_loop():
    env = gym.make('CartPole-v1')

    nn = models.MLPPolicy(env)
    optimizer = torch.optim.Adam(nn.parameters(), lr=0.01)
    updater = PGUpdater(optimizer, gamma=.99)
    policy = NNPolicy(nn, updater)

    train_loop(env=env, policy=policy, n_episodes=1, episode_len=5)


def test_conv_policy():
    env = gym.make('BreakoutDeterministic-v4')
    env.seed(1)
    torch.manual_seed(1)

    nn = models.ConvPolicy(env)
    optimizer = torch.optim.Adam(nn.parameters(), lr=0.01)
    updater = PGUpdater(optimizer, gamma=.99)
    policy = NNPolicy(nn, updater)

    train_loop(env=env, policy=policy, n_episodes=1, episode_len=5)
