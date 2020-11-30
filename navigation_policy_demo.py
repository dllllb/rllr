import gym
import torch

from navigation_models import ConvNavPolicy
from pgrad import PGUpdater, PGUMultyEpisodesUpdater
from policy import RandomActionPolicy, NNPolicy, CategoricalActionPolicy, NNExplorationPolicy
from statesearch import TrajectoryExplorer, generate_train_trajectories, NavigationTrainer, ssim_dist, ssim_l1_dist, ssim_l1_sparce_dist
from constants import *
from actor_critic import ACMultyEpisodesUpdater

from gym_minigrid.wrappers import *
from custom_envs import *
from custom_wrappers import *
from torch.distributions import Categorical
import pickle
import os
import warnings
from constants import *

#env = gym.make('MiniGrid-MyEmptyRandomPos-8x8-v0')
env = gym.make('MiniGrid-MyEmptyRandomPosMetaAction-8x8-v0')
env = RGBImgAndStateObsWrapper(env)

if not os.path.exists(f'./{TASKS_FILE}'):
    explore_policy = CategoricalActionPolicy(sampler=Categorical(probs=torch.tensor([0.2, 0.2, 0.6])))
    te = TrajectoryExplorer(env, explore_policy, n_steps=150, n_episodes=1000)
    tasks = generate_train_trajectories(te, n_initial_points=3, take_prob=.5)
    with open(f'./{TASKS_FILE}', 'wb') as wfs:
        pickle.dump(tasks, wfs)
else:
    with open(f'./{TASKS_FILE}', 'rb') as rfs:
        tasks = pickle.load(rfs)
    warnings.warn('using casched dataset for navigation policy task')
    

nav_nn = ConvNavPolicy(env)
nav_nn.to(DEVICE)
np_optimizer = torch.optim.Adam(nav_nn.parameters(), lr=L_RATE)
lr_scheduler = torch.optim.lr_scheduler.StepLR(np_optimizer, 
                                               step_size=1,
                                               gamma=0.99)
np_updater = PGUMultyEpisodesUpdater(np_optimizer, gamma=.99)
policy = NNExplorationPolicy(nav_nn, np_updater)

nt = NavigationTrainer(env, policy, n_steps_per_episode=BATCH_SIZE, 
                                    n_trials_per_task=1, 
                                    n_actions_without_reward=10,
                                    state_dist=ssim_l1_dist)

for epoch in range(EPOCHS):
    nt(tasks, epoch)
    lr_scheduler.step()

