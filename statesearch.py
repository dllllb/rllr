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
            wm: wmpolicy.WorldModel,
            n_steps: int,
            n_episodes: int):
        self.env = env
        self.exploration_policy = exploration_policy
        self.wm = wm
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
                action = self.exploration_policy(state)
                self.wm(state, action)
                state, reward, done, _ = self.env.step(action)
                trajectory.append(action)
                self.exploration_policy.update(reward)
                self.wm.update(state, reward)
                if done:
                    break

            if not done:
                log.append((initial_trajectory, trajectory, state))

            self.exploration_policy.end_episode()
            self.wm.end_episode()

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
            wm: wmpolicy.WorldModel,
            n_steps_per_episode,
            n_trials_per_task):
        self.env = env
        self.navigation_policy = navigation_policy
        self.wm = wm
        self.n_steps_per_episode = n_steps_per_episode
        self.n_trials_per_task = n_trials_per_task

    def __call__(self, tasks):
        n_steps = 0
        for initial_trajectory, known_trajectory, desired_state in tasks:
            for _ in range(self.n_trials_per_task):
                n_steps += 1
                reward = 0
                for j in range(len(known_trajectory)*3):
                    state = self.env.reset()
                    if initial_trajectory is not None:
                        for a in initial_trajectory:
                            state, reward, done, _ = self.env.step(a)

                    action = self.navigation_policy(state)
                    self.wm(state, action)
                    state, _, done, _ = self.env.step(action)
                    self.wm.update(state, reward)

                    if np.array_equal(state, desired_state):
                        reward = 1
                        break
                    elif done:
                        break

                self.navigation_policy.update(reward)
                if n_steps > self.n_steps_per_episode:
                    n_steps = 0
                    self.navigation_policy.end_episode()
                    self.wm.end_episode()


def test_train_navigation_policy():
    env = gym.make('CartPole-v1')
    explore_policy = RandomActionPolicy(env)

    nav_nn = pgrad.MLPPolicy(env)
    np_optimizer = torch.optim.Adam(nav_nn.parameters(), lr=0.01)
    navigation_policy = pgrad.NNPolicy(nav_nn, pgrad.PGUpdater(np_optimizer, gamma=.99))

    embed_size = 32
    encoder_nn = wmpolicy.mlp_encoder(env, embed_size)
    sp_nn = wmpolicy.MLPNextStatePred(encoder_nn, embed_size)
    rp_nn = wmpolicy.MLPRewardPred(encoder_nn, embed_size)

    wm_params = list(sp_nn.parameters()) + list(rp_nn.parameters()) + list(encoder_nn.parameters())
    wm_optimizer = torch.optim.Adam(wm_params, lr=0.01)
    wm_updater = wmpolicy.WMUpdater(wm_optimizer)

    wm = wmpolicy.NNWorldModel(sp_nn, rp_nn, encoder_nn, wm_updater)

    te = TrajectoryExplorer(env, explore_policy, wm, 5, 2)
    nt = NavigationTrainer(env, navigation_policy, wm, n_trials_per_task=5, n_steps_per_episode=5)

    tasks = generate_train_trajectories(te, 3, .5)
    nt(tasks)
