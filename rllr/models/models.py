import torch
from torch import nn as nn

from ..utils import convert_to_torch


class StateDistanceNetwork(nn.Module):
    """
    State-Embedding model
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


class EncoderDistance:
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
