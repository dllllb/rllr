import torch
from torch import nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pylab as plt

from ..utils import convert_to_torch


class InverseDynamicsModel(nn.Module):
    """
    inverse dynamics model as done by Pathak et al. (2017).
    """
    def __init__(self, encoder, action_size, config):
        super().__init__()
        self.encoder = encoder

        self.fc = nn.Sequential(
            nn.Linear(2 * encoder.output_size, config['hidden_size']),
            nn.ReLU(inplace=False),
            nn.Linear(config['hidden_size'], action_size),
            nn.LogSoftmax(dim=-1)
        )
        self.output_size = action_size

    def forward(self, state, next_state):
        x = self.encoder(state)
        y = self.encoder(next_state)

        return self.fc(torch.cat((x, y), 1))


class StateEmbedder:
    def __init__(self, encoder, device):
        self.encoder = encoder.to(device)
        self.device = device

    def __call__(self, state):
        if isinstance(state, dict):
            state = state['state']
        return self.encoder(convert_to_torch([state], device=self.device)).squeeze(0)


class SameStatesCriterion:
    def __init__(self, encoder, device, threshold=5):
        self.encoder = encoder.to(device)
        self.device = device
        self.threshold = threshold

    def __call__(self, state, goal_state):
        if isinstance(state, dict):
            state, goal_state = state['image'], goal_state['image']
        with torch.no_grad():
            embeds = self.encoder(convert_to_torch([state, goal_state], device=self.device))
        return torch.dist(embeds[0], embeds[1], 2).cpu().item() < self.threshold


class SSIMCriterion:
    def __init__(self, ssim_network, device, threshold=0.5):
        self.ssim = ssim_network.to(device)
        self.device = device
        self.threshold = threshold

    def __call__(self, state, goal_state):
        if isinstance(state, dict):
            state, goal_state = state['image'], goal_state['image']

        with torch.no_grad():
            s = convert_to_torch([state, goal_state], device=self.device)
            dist = 1 - self.ssim(s[:1], s[1:]).cpu().item()
        return dist < self.threshold


class StateSimilarityNetwork(nn.Module):
    def __init__(self, encoder, hidden_size):
        super(StateSimilarityNetwork, self).__init__()

        self.encoder = encoder

        fc = []
        if type(hidden_size) == int:
            fc += [
                nn.Linear(2 * encoder.output_size, hidden_size),
                nn.ReLU(inplace=False),
                nn.Linear(hidden_size, 1)
            ]
        elif type(hidden_size) == list:
            input_size = 2 * encoder.output_size
            for hs in hidden_size:
                fc.append(nn.Linear(input_size, hs))
                fc.append(nn.ReLU(inplace=False))
                input_size = hs
            fc.append(nn.Linear(input_size, 1))
        self.fc = nn.Sequential(*fc)

    def forward(self, x, y):
        x = self.encoder(x)
        y = self.encoder(y)
        z = torch.cat((x, y), 1)
        z = self.fc(z)
        return torch.sigmoid(z)
