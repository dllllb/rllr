import numpy as np
import torch
from torch.nn.functional import smooth_l1_loss

from learner import Updater


class ACUpdater(Updater):
    def __init__(self, optimizer, gamma: float):
        self.optimizer = optimizer
        self.gamma = gamma

    def __call__(self, context_hist, state_hist, reward_hist):
        R = 0
        rewards = []
        policy_losses = []
        value_losses = []

        for r in reward_hist[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)

        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

        for (log_prob, value), r in zip(context_hist, rewards):
            reward = r - value
            policy_losses.append(-log_prob * reward)
            value_losses.append(smooth_l1_loss(torch.tensor(value), torch.tensor(r)))

        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
