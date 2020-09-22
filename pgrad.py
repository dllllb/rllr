import numpy as np
import torch
from learner import Updater
from constants import *


class PGUpdater(Updater):
    def __init__(self, optimizer, gamma: float):
        self.optimizer = optimizer
        self.gamma = gamma
        
    def __call__(self, context_hist, state_hist, reward_hist):
        R = 0
        rewards = []

        for r in reward_hist[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)

        rewards = torch.FloatTensor(rewards)
        #if rewards.size(0) > 1: # or we get nan values
        #    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

        loss = []
        for log_prob, reward in zip(context_hist, rewards):
            loss.append((-log_prob * reward).view(1))
        loss = torch.cat(loss).sum()
        print(f'        LOSS: {round(loss.item()/ len(rewards), 2)}')     

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

