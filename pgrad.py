import numpy as np
import torch
import torch.nn as nn
from learner import Updater, BufferedLearner
from policy import Policy


class NNPolicy(BufferedLearner, Policy):
    def __init__(self, model: nn.Module, updater: Updater):
        super().__init__(updater)
        self.model = model

    def __call__(self, state):
        c = self.model(state)
        action = c.sample()
        return action.item(), c.log_prob(action).view(1)


class PGUpdater(Updater):
    def __init__(self, optimizer, gamma: float):
        self.optimizer = optimizer
        self.gamma = gamma
        
    def __call__(self, context_hist, state_hist, reward_hist):
        R = 0
        rewards = []

        # Discount future rewards back to the present using gamma
        for r in reward_hist[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)

        # Scale rewards
        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

        # Calculate loss
        loss = []
        for log_prob, reward in zip(context_hist, rewards):
            loss.append((-log_prob * reward).view(1))
        loss = torch.cat(loss).sum()

        # Update network weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

