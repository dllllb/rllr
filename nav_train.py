import gym
import torch

import pgrad
from policy import RandomActionPolicy
from statesearch import TrajectoryExplorer, generate_train_trajectories, NavigationTrainer, ssim_dist

env = gym.make('BreakoutDeterministic-v4')
env.seed(1)
torch.manual_seed(1)

explore_policy = RandomActionPolicy(env)

te = TrajectoryExplorer(env, explore_policy, n_episodes=500, n_steps=3)
tasks = generate_train_trajectories(te, 3, .5)

nav_nn = pgrad.ConvPolicy(env)

np_optimizer = torch.optim.Adam(nav_nn.parameters(), lr=0.01)
np_updater = pgrad.PGUpdater(np_optimizer, gamma=.99)
policy = pgrad.NNPolicy(nav_nn, np_updater)

nt = NavigationTrainer(env, policy, n_steps_per_episode=10, state_dist=ssim_dist)
nt(tasks)
