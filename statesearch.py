import warnings

import random
import math
import gym
import numpy as np
import torch
from skimage.measure import compare_ssim

from navigation_models import ConvNavPolicy
from pgrad import PGUpdater
from policy import RandomActionPolicy, Policy, NNPolicy
import time
from torch.distributions import Categorical

import matplotlib.pyplot as plt
from collections import defaultdict
import cv2
import warnings
import tqdm
from constants import *


class TrajectoryExplorer:
    def __init__(
            self,
            env: gym.Env,
            exploration_policy: Policy,
            n_steps: int,
            n_episodes: int):
        self.env = env
        self.exploration_policy = exploration_policy
        self.n_steps = n_steps
        self.n_episodes = n_episodes

    def __call__(self, initial_trajectory=None):
        log = []
        for i in tqdm.tqdm(range(self.n_episodes), desc='tragectories generation...'):
            state = self.env.reset()
            if initial_trajectory is not None:
                for a in initial_trajectory:
                    state, reward, done, _ = self.env.step(a)

            trajectory = list()
            done = False
            initial_state = state
            n_steps = random.randint(int(0.7*self.n_steps), int(1.3*self.n_steps))
            for step in range(self.n_steps):
                action, context = self.exploration_policy(state)
                state, reward, done, _ = self.env.step(action)
                trajectory.append(action)
                self.exploration_policy.update(context, state, reward)
                if done:
                    break
                
                if ssim_l1_dist(initial_state, state) > 10 and step > 0.5*self.n_steps:
                    warnings.warn('hard coded break exploration for test in TrajectoryExplorer')
                    break 

            if not done:
                log.append((initial_trajectory, trajectory, state))

            self.exploration_policy.end_episode()

        return log


def generate_train_trajectories(trajectory_explorer: TrajectoryExplorer, n_initial_points: int, take_prob: float):
    initial_trajectory = list()
    tasks = list()
    for _ in range(n_initial_points):
        trajectories = trajectory_explorer(initial_trajectory)
        selected_idx = np.random.choice(len(trajectories), int(len(trajectories) * take_prob))
        selected = [trajectories[i] for i in selected_idx]
        tasks.extend(selected)
        initial_trajectory = selected[np.random.choice(len(selected), 1)[0]][0]

    return tasks


def ssim_dist(state, target):
    return (compare_ssim(state, target, multichannel=True)+1)/2

def ssim_l1_dist(state, target):
    srs, dst = state['agent_pos'], target['agent_pos']
    delta = abs(srs[0] - dst[0]) + abs(srs[1] - dst[1])
    return delta

def mse_dist(state, target):
    return ((state - target)**2).mean()


class NavigationTrainer:
    def __init__(
            self,
            env: gym.Env,
            navigation_policy: Policy,
            n_steps_per_episode=3,
            n_trials_per_task=3,
            n_actions_without_reward=30,
            state_dist=compare_ssim,
            render=False,
            show_task=False):
        self.env = env
        self.navigation_policy = navigation_policy
        self.n_steps_per_episode = n_steps_per_episode
        self.n_trials_per_task = n_trials_per_task
        self.n_actions_without_reward = n_actions_without_reward
        self.state_dist = state_dist
        self.visualize = render
        self.show_task = show_task
        self.n_steps = 0
        self.completed_tasks = 0

    def render(self):
        if self.visualize:
            self.env.render()

    def plt_show(self, states, names):
        if self.show_task and self.visualize:
            fig=plt.figure(figsize=(4, 4))
            for i, state in enumerate(states):
                fig.add_subplot(1, len(states), i+1)
                plt.imshow(state['image'])
                plt.title(names[i])

            plt.show()
            input('\npress any key to start new task ...\n')
            plt.close()

    def __call__(self, tasks, epoch):
        actions_description = {0: "r", 1: "d", 2: "l", 3: "u"}
        running_reward = list()
        with tqdm.tqdm(total=len(tasks), desc=f'epoch {epoch}') as pbar:
            for i, (initial_trajectory, known_trajectory, desired_state) in enumerate(tasks):

                for j in range(self.n_trials_per_task):

                    state = self.env.reset()
                    if initial_trajectory is not None:
                        for a in initial_trajectory:
                            self.render()
                            state, reward, done, _ = self.env.step(a)

                    initial_dist = min_dist = self.state_dist(state, desired_state)
                    if initial_dist == 0 or \
                       state['agent_pos'][0] == desired_state['agent_pos'][0] or\
                       state['agent_pos'][1] == desired_state['agent_pos'][1]:
                        continue

                    no_reward_actions = 0
                    positive_reward = 0
                    initial_state = state

                    trajectory_rewards = ''
                    prev_dist = None
                    make_actions = [('|', '')]
                    for _ in range(len(known_trajectory)*100):
                        action, context = self.navigation_policy((state, desired_state))
                        # track action distributions
                        if make_actions[-1][0] == actions_description[action]:
                            make_actions[-1] = (make_actions[-1][0], make_actions[-1][1]+1)
                        else:
                            make_actions.append((actions_description[action], 1))

                        self.render()
                        next_state, _, done, _ = self.env.step(action)

                        curent_dist = self.state_dist(next_state, desired_state)
                        if prev_dist == None: prev_dist = curent_dist

                        if curent_dist < min_dist:# or curent_dist < prev_dist:
                            min_dist = min(curent_dist, min_dist)
                            no_reward_actions = max(no_reward_actions - 1, 0)
                            positive_reward += 1

                            price = random.random() if random.random() > 0.7 else 0 # stohastic semisparce rewards
                            #price = 0 if min_dist > 0 else 1 # sparce mode
                            #price = 1 # dense mode
                            self.navigation_policy.update(context, (state, desired_state), price)
                            trajectory_rewards += '+'
                            state = next_state
                            if min_dist == 0:
                                self.completed_tasks += 1
                                break
                        elif done:
                            self.navigation_policy.update(context, (state, desired_state), -1)
                            state = next_state
                            break
                        else:
                            trajectory_rewards += '-'
                            no_reward_actions += 1
                            if no_reward_actions > self.n_actions_without_reward:
                                self.navigation_policy.update(context, (state, desired_state), -1)
                                state = next_state
                                break
                            else:
                                self.navigation_policy.update(context, (state, desired_state), 0)
                                state = next_state
                        prev_dist = curent_dist

                    running_reward.append(positive_reward)

                    self.navigation_policy.end_episode()
                    self.n_steps += 1
                    if self.n_steps % self.n_steps_per_episode == self.n_steps_per_episode - 1:
                        #completed_accuracy = round(float(self.completed_tasks)/self.n_steps_per_episode, 2)
                        self.navigation_policy.end()
                        #torch.save(self.navigation_policy.model, f'./saved_models/pretrained_navigation_model_{self.env.spec.id}.pkl')
                        actions_str = ' '.join([f'{act}{num}' for act, num in make_actions]) + '|'
                        pbar.set_postfix({
                            'trajectory': trajectory_rewards + ' '* max(30 - len(trajectory_rewards), 1) + '-',
                            'made actions': actions_str + ' '* max(40 - len(actions_str), 1) + '-',
                            'completed': f'{self.completed_tasks}/{self.n_steps_per_episode}',
                            'H': round(ENTROPY_WEIGHT(), 2),
                            'mean reward': np.array(running_reward).mean()
                            })
                        self.completed_tasks = 0
                    self.plt_show([initial_state, state, desired_state], ['initial state', 'state', 'desired state'])

                pbar.update()


def test_train_navigation_policy_ssim():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        env = gym.make('BreakoutDeterministic-v4')
    env.seed(1)
    torch.manual_seed(1)

    explore_policy = RandomActionPolicy(env)

    nav_nn = ConvNavPolicy(env)
    np_optimizer = torch.optim.Adam(nav_nn.parameters(), lr=0.01)
    np_updater = PGUpdater(np_optimizer, gamma=.99)
    policy = NNPolicy(nav_nn, np_updater)

    te = TrajectoryExplorer(env, explore_policy, 5, 2)
    tasks = generate_train_trajectories(te, 3, .5)

    nt = NavigationTrainer(env, policy, n_steps_per_episode=3, state_dist=ssim_dist)
    nt(tasks)
