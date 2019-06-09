import gym
import numpy as np


class LearnablePolicy:
    def update(self, reward):
        pass

    def __call__(self, state):
        """
        :param state: current state of the environment
        :return: proposed action
        """
        pass


class RandomActionPolicy(LearnablePolicy):
    def __init__(self, env):
        self.actions = env.action_space

    def __call__(self, _):
        return self.actions.sample()


class WorldModel:
    def update(self, prev_state, action, next_state, reward):
        pass

    def __call__(self, state, action):
        """
        :param state: current state of the environment
        :param action: proposed action
        :return: state embedding and predicted reward
        """
        pass


class TrajectoryExplorer:
    def __init__(self, env: gym.Env, exploration_policy, wm, n_steps, n_episodes):
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
                prev_state = state
                state, reward, done, _ = self.env.step(action)
                trajectory.append(action)
                self.exploration_policy.update(reward)
                self.wm.update(prev_state, action, state, reward)
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


class NavigationTrainer:
    def __init__(self, env: gym.Env, navigation_policy, wm):
        self.env = env
        self.navigation_policy = navigation_policy
        self.wm = wm

    def __call__(self, tasks, n_trials_per_task):
        for initial_trajectory, known_trajectory, desired_state in tasks:
            for _ in range(n_trials_per_task):
                reward = 0
                for _ in range(len(known_trajectory)*3):
                    state = self.env.reset()
                    if initial_trajectory is not None:
                        for a in initial_trajectory:
                            state, reward, done, _ = self.env.step(a)

                    action = self.navigation_policy(state)
                    prev_state = state
                    state, _, done, _ = self.env.step(action)
                    self.wm.update(prev_state, action, state, reward)
                    if np.array_equal(state, desired_state):
                        reward = 1
                        break
                    elif done:
                        break

                self.navigation_policy.update(reward)


def test_train_navigation_policy():
    env = gym.make('CartPole-v1')
    explore_policy = RandomActionPolicy(env)
    navigation_policy = RandomActionPolicy(env)
    wm = WorldModel()

    te = TrajectoryExplorer(env, explore_policy, wm, 5, 2)
    nt = NavigationTrainer(env, navigation_policy, wm)

    tasks = generate_train_trajectories(te, 3, .5)
    nt(tasks, n_trials_per_task=5)


def test_trajectory_explorer():
    env = gym.make('CartPole-v1')
    explore_policy = RandomActionPolicy(env)
    wm = WorldModel()
    te = TrajectoryExplorer(env, explore_policy, wm, 5, 1)
    te()
