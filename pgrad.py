import numpy as np
import torch
from learner import Updater, MultyEpisodesUpdater
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
        if rewards.size(0) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

        loss = []
        for log_prob, reward in zip(context_hist, rewards):
            loss.append((-log_prob * reward).view(1))
        loss = torch.cat(loss).sum()   

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class PGUMultyEpisodesUpdater(MultyEpisodesUpdater):
    def __init__(self, optimizer, gamma: float):
        self.optimizer = optimizer
        self.gamma = gamma

    def calc_episode(self, context_hist, state_hist, reward_hist):
        R = 0
        rewards = []

        for r in reward_hist[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)

        rewards = torch.FloatTensor(rewards)

        loss = []
        for (action, (log_prob, entropy)), reward in zip(context_hist, rewards):
            # everage entropy in episode
            loss.append((-log_prob * reward - entropy*ENTROPY_WEIGHT()/len(rewards)).view(1))
        loss = torch.cat(loss).sum()

        return loss
        
    def __call__(self, episodes):
        episode_losses = []
        for episode in episodes:
            episode_losses.append(self.calc_episode(episode.context_buffer, episode.state_buffer, episode.reward_buffer).view(1, -1))

        loss = torch.cat(episode_losses, 0).mean()
        ENTROPY_WEIGHT.step()

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
