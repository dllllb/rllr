import gym
import numpy as np


class ExplorationStrategy:
    def update(self, reward):
        pass

    def next_action(self, state):
        pass


class RandomExplorationStrategy(ExplorationStrategy):
    def __init__(self, env):
        self.actions = env.action_space

    def next_action(self, _):
        return self.actions.sample()


class TrajectoryExplorer:
    def __init__(self, env: gym.Env, policy, n_steps, n_episodes):
        self.env = env
        self.policy = policy
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
                action = self.policy.next_action(state)
                trajectory.append(action)
                state, reward, done, _ = self.env.step(action)
                self.policy.update(reward)
                if done:
                    break

            if not done:
                log.append((initial_trajectory, trajectory, state))

        return log


def generate_train_trajectories(trajectory_explorer, n_initial_points, take_prob):
    initial_trajectory = None
    tasks = list()
    for _ in range(n_initial_points):
        trajectories = trajectory_explorer(initial_trajectory)
        selected_idx = np.random.choice(len(trajectories), int(len(trajectories) * take_prob))
        selected = [trajectories[i] for i in selected_idx]
        tasks.extend(selected)
        initial_trajectory = selected[np.random.choice(len(selected), 1)[0]][0]

    return tasks


def train_navigation_policy(env: gym.Env, policy, tasks, n_trials_per_task):
    for initial_trajectory, known_trajectory, desired_state in tasks:
        for _ in range(n_trials_per_task):
            reward = 0
            for _ in range(len(known_trajectory)*3):
                state = env.reset()
                if initial_trajectory is not None:
                    for a in initial_trajectory:
                        state, reward, done, _ = env.step(a)

                action = policy.next_action(state)
                state, _, done, _ = env.step(action)
                if np.array_equal(state, desired_state):
                    reward = 1
                    break
                elif done:
                    break

            policy.update(reward)


def test_train_navigation_policy():
    env = gym.make('CartPole-v1')
    explore_policy = RandomExplorationStrategy(env)
    navigation_policy = RandomExplorationStrategy(env)

    te = TrajectoryExplorer(env, explore_policy, 5, 2)
    tasks = generate_train_trajectories(te, 3, .5)

    train_navigation_policy(env, navigation_policy, tasks, n_trials_per_task=5)


def test_trajectory_explorer():
    env = gym.make('CartPole-v1')
    explore_policy = RandomExplorationStrategy(env)
    te = TrajectoryExplorer(env, explore_policy, 5, 1)
    te()
