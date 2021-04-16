
import numpy as np
import gym

import torch
import torch.nn as nn

import train_worker
from rllr.env import NavigationGoalWrapper
from rllr.env.gym_minigrid_navigation import environments as minigrid_envs

from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GoalStateExtenderWrapper(NavigationGoalWrapper):
    """
    Wrapper return state and goal_state_concatenated
    """

    def __init__(self, env):
        super().__init__(env, env.goal_achieving_criterion)
        obs_space = env.observation_space['image']
        low = np.expand_dims(obs_space.low, axis=0).repeat(2, 0)
        high = np.expand_dims(obs_space.high, axis=0).repeat(2, 0)
        extended_shape = (2,) + obs_space.shape
        self.observation_space = gym.spaces.Box(low, high, extended_shape)
        self.action_space = gym.spaces.Discrete(3)

    def _extend_state(self, state):
        return np.array([state["image"], self.env.goal_state['image']])

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return self._extend_state(next_state), reward, done, info

    def reset(self):
        state = super().reset()
        return self._extend_state(state)


class StateExtenderWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space['image']
        low = np.expand_dims(obs_space.low, axis=0).repeat(2, 0)
        high = np.expand_dims(obs_space.high, axis=0).repeat(2, 0)
        extended_shape = (2,) + obs_space.shape
        self.observation_space = gym.spaces.Box(low, high, extended_shape)
        self.action_space = gym.spaces.Discrete(3)

    def _extend_state(self, state):
        return np.array([state["image"], state["image"]])

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return self._extend_state(next_state), reward, done, info

    def reset(self):
        state = super().reset()
        return self._extend_state(state)


class ExtendedStateFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Tuple,
                 features_dim: int = 64, hidden: int = 64):
        super(ExtendedStateFeatureExtractor, self).__init__(observation_space, 2*features_dim)

        k, n = observation_space.shape[1] - 2, observation_space.shape[-1]
        input_dim = k * k * n

        self.state_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, features_dim),
            nn.Tanh()
        )

        self.goal_state_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, features_dim),
            nn.Tanh()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x1 = observations[:, 0, :]
        x1 = x1[:, 1:-1, 1:-1].flatten(start_dim=1)
        x1 = self.state_encoder(x1)

        x2 = observations[:, 1, :]
        x2 = x2[:, 1:-1, 1:-1].flatten(start_dim=1)
        x2 = self.goal_state_encoder(x2)

        return torch.cat([x1, x2], dim=1)


def train_1st_stage():

    worker_env_config = {
        "env_type": "gym_minigrid",
        "env_task": "MiniGrid-Empty",
        "grid_size": 8,
        "action_size": 3,
        "rgb_image": False,
        "goal_achieving_criterion": "position",
        "goal_type": "random",
        "video_path": "outputs/video/"
    }
    worker_env = train_worker.gen_navigation_env(worker_env_config)
    worker_env = GoalStateExtenderWrapper(worker_env)

    policy_kwargs = dict(
        features_extractor_class=ExtendedStateFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=64, hidden=128),
        net_arch=[128, 128]
    )

    dqn_kwargs = {
        "learning_rate": 1e-3,
        "buffer_size": int(1e5),
        "batch_size": 128,
        "learning_starts": 1000,
        "exploration_fraction": 0.995,
        "gamma": 0.9,
        "tau": 0.999,
        "seed": 42,
        "policy_kwargs": policy_kwargs,
        "verbose": 1
    }

    agent = DQN("MlpPolicy", worker_env, **dqn_kwargs)
    agent.learn(100000, log_interval=10)
    return agent


def train_2nd_stage(agent):

    master_env_config = {
        "env_type": "gym_minigrid",
        "env_task": "MiniGrid-Empty",
        "grid_size": 8,
        "action_size": 3,
        "rgb_image": False,
        "video_path": "outputs/video/"
    }

    master_env = minigrid_envs.gen_wrapped_env(master_env_config)
    master_env = StateExtenderWrapper(master_env)
    agent.set_env(master_env)

    for param in agent.policy.q_net.features_extractor.state_encoder.parameters():
        param.requires_gradient = False

    for param in agent.policy.q_net.q_net.parameters():
        param.requires_gradient = False

    agent.learn(100000, log_interval=10)


if __name__ == '__main__':
    agent = train_1st_stage()
    train_2nd_stage(agent)