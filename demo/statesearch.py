import warnings

import gym
import numpy as np
import torch
from skimage.measure import compare_ssim

from navigation_models import ConvNavPolicy
from pgrad import PGUpdater
from policy import RandomActionPolicy, Policy, NNPolicy


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


def mse_dist(state, target):
    return ((state - target)**2).mean()


class NavigationTrainer:
    def __init__(
            self,
            env: gym.Env,
            navigation_policy: Policy,
            n_steps_per_episode,
            n_trials_per_task=3,
            n_actions_without_reward=20,
            state_dist=compare_ssim):
        self.env = env
        self.navigation_policy = navigation_policy
        self.n_steps_per_episode = n_steps_per_episode
        self.n_trials_per_task = n_trials_per_task
        self.n_actions_without_reward = n_actions_without_reward
        self.state_dist = state_dist

    def __call__(self, tasks):
        n_steps = 0
        running_reward = list()
        for i, (initial_trajectory, known_trajectory, desired_state) in enumerate(tasks):
            for _ in range(self.n_trials_per_task):
                n_steps += 1

                state = self.env.reset()
                if initial_trajectory is not None:
                    for a in initial_trajectory:
                        state, reward, done, _ = self.env.step(a)

                max_sim = self.state_dist(state, desired_state)
                no_reward_actions = 0
                positive_reward = 0

                for j in range(len(known_trajectory)*100):
                    action, context = self.navigation_policy((state, desired_state))
                    state, _, done, _ = self.env.step(action)

                    sim = self.state_dist(state, desired_state)
                    if sim > max_sim:
                        max_sim = sim
                        no_reward_actions = 0
                        positive_reward += 1
                        self.navigation_policy.update(context, (state, desired_state), 1)
                        if max_sim > 0.999:
                            break
                    elif done:
                        self.navigation_policy.update(context, (state, desired_state), -1)
                        break
                    else:
                        no_reward_actions += 1
                        self.navigation_policy.update(context, (state, desired_state), 0)
                        if no_reward_actions > self.n_actions_without_reward:
                            break

                running_reward.append(positive_reward)

                if n_steps > self.n_steps_per_episode:
                    self.navigation_policy.end_episode()

            if i % 10 == 0:
                print(f'tasks processed: {i}, mean reward: {np.array(running_reward).mean()}')


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