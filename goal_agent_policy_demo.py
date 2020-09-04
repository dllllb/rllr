import gym
import torch

from navigation_models import ConvNavPolicy, GoalConvNavPolicy
from pgrad import PGUpdater
from policy import NNPolicy
from train import train_loop

env = gym.make('BreakoutDeterministic-v4')
env.seed(1)
torch.manual_seed(1)

nav_nn = torch.load('./saved_models/pretrained_navigation_model.pkl')
# freeze slave agent
for param in nav_nn.conv.parameters():
    param.requires_grad = False
model = GoalConvNavPolicy(model=nav_nn, env=env, starter=None)

np_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
np_updater = PGUpdater(np_optimizer, gamma=.99)
policy = NNPolicy(model, np_updater)

train_loop(env, policy, n_episodes=10, episode_len=1000, render=False, seed=1)
