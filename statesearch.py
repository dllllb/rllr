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

import matplotlib.pyplot as plt
from collections import defaultdict
import cv2


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
        for i in range(self.n_episodes):
            state = self.env.reset()
            if initial_trajectory is not None:
                for a in initial_trajectory:
                    state, reward, done, _ = self.env.step(a)

            trajectory = list()
            done = False
            n_steps = random.randint(int(0.7*self.n_steps), int(1.3*self.n_steps))
            for _ in range(self.n_steps):
                action, context = self.exploration_policy(state)
                state, reward, done, _ = self.env.step(action)
                trajectory.append(action)
                self.exploration_policy.update(context, state, reward)
                if done:
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

def ssim_dist_euclidian(state, target):
    p0, p1 = state['agent_pos'], target['agent_pos']
    return math.exp(-math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2))



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

    def render(self):
        if self.visualize:
            self.env.render()
            #time.sleep(1.0/12)


    def plt_show(self, state, disired_state):
        if self.show_task and self.visualize:
            fig=plt.figure(figsize=(4, 4))
            #fig=plt.figure()

            if state is not None:
                fig.add_subplot(1, 2, 1)
                plt.imshow(state['image'])

            if disired_state is not None:
                fig.add_subplot(1, 2, 2)
                plt.imshow(disired_state['image'])

            plt.show()
            input('...')
            plt.close()

    def __call__(self, tasks):
        n_steps = 0
        running_reward = list()
        for i, (initial_trajectory, known_trajectory, desired_state) in enumerate(tasks):
            print('\n--------------------------------------------')
            print(f'NEW TASK {i}:')
            #self.plt_show(None, desired_state)

            for _ in range(self.n_trials_per_task):
                print(f'    trial {_+1}/{self.n_trials_per_task}:')
                task_actions = defaultdict(int)

                n_steps += 1

                state = self.env.reset()
                if initial_trajectory is not None:
                    for a in initial_trajectory:
                        self.render()
                        state, reward, done, _ = self.env.step(a)

                max_sim = self.state_dist(state, desired_state)
                print(f'        initial simularity {max_sim}')
                no_reward_actions = 0
                positive_reward = 0

                for j in range(len(known_trajectory)*100):
                    if isinstance(state, np.ndarray):
                        action, context = self.navigation_policy((state, desired_state))
                    elif isinstance(state, dict):
                        action, context = self.navigation_policy((state['image'], desired_state['image']))
                    else:
                        raise AttributeError(f'unsupported state type: {type(state)}')

                    task_actions[action] += 1
                    self.render()

                    state, _, done, _ = self.env.step(action)

                    sim = self.state_dist(state, desired_state)
                    if sim > max_sim:
                        max_sim = sim
                        no_reward_actions = 0
                        positive_reward += 1
                        self.navigation_policy.update(context, (state, desired_state), 1)
                        print('        +')
                        if max_sim > 0.999:
                            print(f'        ***COMPLETED***')
                            break
                    elif done:
                        self.navigation_policy.update(context, (state, desired_state), -1)
                        #print(f'        env done')
                        break
                    else:
                        if sim != max_sim:
                            print('        -')
                        no_reward_actions += 1
                        self.navigation_policy.update(context, (state, desired_state), 0)
                        if no_reward_actions > self.n_actions_without_reward:
                            #print(f'        no_reward_actions break')
                            break


                running_reward.append(positive_reward)

                if n_steps > self.n_steps_per_episode:
                    self.navigation_policy.end_episode()
                    n_steps = 0 # it wasnot done before 

                # show extion distibutions
                task_actions = [task_actions[k] for k in sorted(list(task_actions.keys()))]
                #print(f'        trial action distribution: {task_actions}')

                #print(f'        END: trial reward {positive_reward}, best_sim {max_sim}')
                self.plt_show(state, desired_state)

            if i % 10 == 0:
                torch.save(self.navigation_policy.model, f'./saved_models/pretrained_navigation_model_{self.env.spec.id}.pkl')
                #print(f'tasks processed: {i}, mean reward: {np.array(running_reward).mean()}')


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
