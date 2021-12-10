import gym
import torch
from torch.nn import functional as F

from rllr.buffer import ReplayBuffer
from rllr.utils import convert_to_torch


class RandomNetworkDistillationReward(gym.Wrapper):
    """
    Wrapper for adding Random Network Distillation Reward
        [Burda et al. (2019)](https://arxiv.org/abs/1810.12894)
    """
    def __init__(self,
                 env,
                 target,
                 predictor,
                 device,
                 learning_rate=0.001,
                 buffer_size=10000,
                 batch_size=64,
                 update_step=4,
                 mean_gamma=0.99,
                 use_extrinsic_reward=False,
                 gamma=1):

        self.device = device
        self.target = target.to(device)
        self.predictor = predictor.to(device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size, device=device)
        self.steps_count = 0
        self.update_step = update_step
        self.running_mean = 0
        self.running_sd = 1
        self.mean_gamma = mean_gamma
        self.gamma = gamma
        self.alpha = 1
        self.eps = 1e-5
        self.use_extrinsic_reward = use_extrinsic_reward
        super().__init__(env)

    def reset(self):
        state = super().reset()
        return state

    def _learn(self):
        # Sample batch from buffer buffer
        states = list(self.replay_buffer.sample())[0]
        with torch.no_grad():
            targets = self.target(states)
        outputs = self.predictor(states)
        self.optimizer.zero_grad()
        loss = F.mse_loss(outputs, targets)
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            intrinsic_reward = (targets - outputs).abs().pow(2).sum(1)
            self.running_mean = self.mean_gamma * self.running_mean + (1 - self.mean_gamma) * intrinsic_reward.mean()
            self.running_sd = self.mean_gamma * self.running_sd + (1 - self.mean_gamma) * intrinsic_reward.var().pow(0.5)
            self.alpha *= self.gamma

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        if isinstance(state, dict):
            self.replay_buffer.add(state['state'], )
            converted_state = convert_to_torch([state['state']], device=self.device)
        else:
            self.replay_buffer.add(state, )
            converted_state = convert_to_torch([state], device=self.device)

        if self.alpha > self.eps:
            with torch.no_grad():
                intrinsic_reward = (self.target(converted_state) - self.predictor(converted_state)).abs().pow(2).sum()
                intrinsic_reward = (intrinsic_reward - self.running_mean) / self.running_sd
                intrinsic_reward = torch.clamp(intrinsic_reward, max=5, min=-5).item()

            self.steps_count = (self.steps_count + 1) % self.update_step
            if self.steps_count == 0:
                if self.replay_buffer.is_enough():
                    self._learn()
        else:
            intrinsic_reward = 0

        return state, reward * self.use_extrinsic_reward + self.alpha * intrinsic_reward, done, info