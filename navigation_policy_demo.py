import gym
import torch

from navigation_models import ConvNavPolicy
from pgrad import PGUpdater
from policy import RandomActionPolicy, NNPolicy
from statesearch import TrajectoryExplorer, generate_train_trajectories, NavigationTrainer, ssim_dist

env = gym.make('BreakoutDeterministic-v4')
env.seed(1)
torch.manual_seed(1)

explore_policy = RandomActionPolicy(env)

te = TrajectoryExplorer(env, explore_policy, n_steps=10, n_episodes=50)
tasks = generate_train_trajectories(te, n_initial_points=3, take_prob=.5)

nav_nn = ConvNavPolicy(env)
np_optimizer = torch.optim.Adam(nav_nn.parameters(), lr=0.01)
np_updater = PGUpdater(np_optimizer, gamma=.99)
policy = NNPolicy(nav_nn, np_updater)

nt = NavigationTrainer(env, policy, n_steps_per_episode=3, state_dist=ssim_dist)
nt(tasks)
