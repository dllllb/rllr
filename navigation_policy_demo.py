import gym
import torch

from navigation_models import ConvNavPolicy, StateAPINavPolicy
from pgrad import PGUpdater
from policy import RandomActionPolicy, NNPolicy, CategoricalActionPolicy
from statesearch import TrajectoryExplorer, generate_train_trajectories, NavigationTrainer, ssim_dist, ssim_dist_euclidian, ssim_dist_abs
from constants import *

from gym_minigrid.wrappers import *
from custom_envs import *
from custom_wrappers import *
from torch.distributions import Categorical
import pickle
import os
import warnings

# ---------------------------------------------------------------
#env = gym.make('BreakoutDeterministic-v4')

# ---------------------------------------------------------------
#env = gym.make('CartPole-v0')

# ---------------------------------------------------------------
#env = gym.make('MiniGrid-MyEmpty-8x8-v0')
env = gym.make('MiniGrid-MyEmptyRandomPos-8x8-v0')
#env = RGBImgObsWrapper(env) # Get pixel observations
#env = ImgObsWrapper(env) # Get rid of the 'mission' field
env = RGBImgAndStateObsWrapper(env)

#env.seed(1)
#torch.manual_seed(1)

if not os.path.exists('./tasks.pkl'):
    #explore_policy = RandomActionPolicy(env)
    explore_policy = CategoricalActionPolicy(sampler=Categorical(probs=torch.tensor([0.2, 0.2, 0.6])))

    te = TrajectoryExplorer(env, explore_policy, n_steps=150, n_episodes=50)
    tasks = generate_train_trajectories(te, n_initial_points=3, take_prob=.5)
    with open('./tasks.pkl', 'wb') as wfs:
        pickle.dump(tasks, wfs)
else:
    with open('./tasks.pkl', 'rb') as rfs:
        tasks = pickle.load(rfs)
    warnings.warn('using casched dataset for navigation policy task')
    


#nav_nn = ConvNavPolicy(env)
nav_nn = StateAPINavPolicy(env)
nav_nn.to(DEVICE)
np_optimizer = torch.optim.Adam(nav_nn.parameters(), lr=0.01)#lr=3e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(np_optimizer, 
                                               step_size=1, 
                                               gamma=0.9)
np_updater = PGUpdater(np_optimizer, gamma=.99)
policy = NNPolicy(nav_nn, np_updater)

nt = NavigationTrainer(env, policy, n_steps_per_episode=3, 
                                    n_trials_per_task=3, 
                                    n_actions_without_reward=15,
                                    state_dist=ssim_dist_abs,#ssim_dist, 
                                    render=False, 
                                    show_task=False)

EPOCHS = 50
for epoch in range(EPOCHS):
    print(f'================================= EPOCH {epoch+1}/{EPOCHS} =================================')
    nt(tasks)
    lr_scheduler.step()
