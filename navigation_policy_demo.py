import gym
import torch

from navigation_models import ConvNavPolicy
from pgrad import PGUpdater
from policy import RandomActionPolicy, NNPolicy
from statesearch import TrajectoryExplorer, generate_train_trajectories, NavigationTrainer, ssim_dist, ssim_dist_euclidian
from constants import *

from gym_minigrid.wrappers import *
from custom_envs import *
from custom_wrappers import *

# ---------------------------------------------------------------
#env = gym.make('BreakoutDeterministic-v4')

# ---------------------------------------------------------------
#env = gym.make('CartPole-v0')

# ---------------------------------------------------------------
env = gym.make('MiniGrid-MyEmpty-8x8-v0')
#env = RGBImgObsWrapper(env) # Get pixel observations
#env = ImgObsWrapper(env) # Get rid of the 'mission' field
env = RGBImgAndStateObsWrapper(env)

env.seed(1)
torch.manual_seed(1)

explore_policy = RandomActionPolicy(env)

te = TrajectoryExplorer(env, explore_policy, n_steps=50, n_episodes=50)
tasks = generate_train_trajectories(te, n_initial_points=3, take_prob=.5)

nav_nn = ConvNavPolicy(env)
nav_nn.to(DEVICE)
np_optimizer = torch.optim.Adam(nav_nn.parameters(), lr=0.0001)
np_updater = PGUpdater(np_optimizer, gamma=.9)#99)
policy = NNPolicy(nav_nn, np_updater)

nt = NavigationTrainer(env, policy, n_steps_per_episode=0, 
                                    n_trials_per_task=3, 
                                    n_actions_without_reward=50,
                                    state_dist=ssim_dist_euclidian,#ssim_dist, 
                                    render=True, 
                                    show_task=False)
nt(tasks)
