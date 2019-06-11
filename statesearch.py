import gym
import numpy as np
import torch

import pgrad
import wmpolicy
from policy import RandomActionPolicy, Policy


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
    initial_trajectory = None
    tasks = list()
    for _ in range(n_initial_points):
        trajectories = trajectory_explorer(initial_trajectory)
        selected_idx = np.random.choice(len(trajectories), int(len(trajectories) * take_prob))
        selected = [trajectories[i] for i in selected_idx]
        tasks.extend(selected)
        initial_trajectory = selected[np.random.choice(len(selected), 1)[0]][0]

    return tasks


class NavigationTrainer:
    def __init__(
            self,
            env: gym.Env,
            navigation_policy: Policy,
            n_steps_per_episode,
            n_trials_per_task):
        self.env = env
        self.navigation_policy = navigation_policy
        self.n_steps_per_episode = n_steps_per_episode
        self.n_trials_per_task = n_trials_per_task

    def __call__(self, tasks):
        n_steps = 0
        for initial_trajectory, known_trajectory, desired_state in tasks:
            for _ in range(self.n_trials_per_task):
                n_steps += 1
                for j in range(len(known_trajectory)*3):
                    state = self.env.reset()
                    if initial_trajectory is not None:
                        for a in initial_trajectory:
                            state, reward, done, _ = self.env.step(a)

                    action, context = self.navigation_policy(state)
                    state, _, done, _ = self.env.step(action)

                    if np.array_equal(state, desired_state):
                        self.navigation_policy.update(context, state, 1)
                        break
                    elif done:
                        self.navigation_policy.update(context, state, 0)
                        break
                    else:
                        self.navigation_policy.update(context, state, 0)

                if n_steps > self.n_steps_per_episode:
                    n_steps = 0
                    self.navigation_policy.end_episode()


def test_train_navigation_policy():
    env = gym.make('CartPole-v1')
    env.seed(1)
    explore_policy = RandomActionPolicy(env)

    embed_size = 32
    encoder_nn = wmpolicy.mlp_encoder(env, embed_size)
    nav_nn = pgrad.MLPPolicy(env)

    np_optimizer = torch.optim.Adam(nav_nn.parameters(), lr=0.01)
    np_updater = pgrad.PGUpdater(np_optimizer, gamma=.99)
    policy = pgrad.NNPolicy(nav_nn, np_updater)

    te = TrajectoryExplorer(env, explore_policy, 5, 2)
    tasks = generate_train_trajectories(te, 3, .5)

    nt = NavigationTrainer(env, policy, n_trials_per_task=5, n_steps_per_episode=5)
    nt(tasks)

    sp_nn = wmpolicy.MLPNextStatePred(encoder_nn, embed_size)
    sp_optimizer = torch.optim.Adam(list(sp_nn.parameters()) + list(encoder_nn.parameters()), lr=0.01)
    sp_updater = wmpolicy.DistUpdater(sp_optimizer)
    policy = wmpolicy.SPPolicy(sp_nn, nav_nn, encoder_nn, sp_updater)

    nt = NavigationTrainer(env, policy, n_trials_per_task=5, n_steps_per_episode=5)
    nt(tasks)

    rp_nn = wmpolicy.MLPRewardPred(encoder_nn, embed_size)
    mse_optimizer = torch.optim.Adam(list(rp_nn.parameters()) + list(encoder_nn.parameters()), lr=0.01)
    mse_updater = wmpolicy.MSEUpdater(mse_optimizer)
    policy = wmpolicy.RPPolicy(rp_nn, nav_nn, mse_updater)

    nt = NavigationTrainer(env, policy, n_trials_per_task=5, n_steps_per_episode=5)
    nt(tasks)
