import gym
import gym_minigrid

import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper


def _to_torch(arr, device='cpu'):
    if isinstance(arr, dict):
        return {
            key: _to_torch(arr[key], device) for key in arr.keys()
        }
    return torch.from_numpy(arr).to(device)


def make_vec_envs(make_env, num_processes, device):
    envs = [make_env(env_id) for env_id in range(num_processes)]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    envs = VecPyTorch(envs, device)
    return envs


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        self.reward_range = None

    def reset(self):
        return _to_torch(self.venv.reset(), self.device)

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)

        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return _to_torch(obs, self.device), reward, done, info

