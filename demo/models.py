import gym
import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical


class MLPPolicy(nn.Module):
    def __init__(self, env: gym.Env):
        super().__init__()
        state_space = env.observation_space.shape[0]
        action_space = env.action_space.n

        self.model = nn.Sequential(
            nn.Linear(state_space, 128, bias=False),
            nn.Dropout(p=0.6),
            nn.ReLU(),
            nn.Linear(128, action_space, bias=False),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        state = torch.from_numpy(state).type(torch.FloatTensor)
        ap = self.model(state)
        return ap


class MLPPolicyAV(nn.Module):
    def __init__(self, env: gym.Env):
        super().__init__()
        state_space = env.observation_space.shape[0]
        action_space = env.action_space.n

        self.basenet = nn.Sequential(
            nn.Linear(state_space, 128, bias=False),
            nn.Dropout(p=0.6),
            nn.ReLU(),
        )

        self.action_head = nn.Sequential(
            nn.Linear(128, action_space, bias=False),
            nn.Softmax(dim=-1)
        )

        self.value_head = nn.Sequential(
            nn.Linear(128, 1, bias=False),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        state = torch.from_numpy(state).type(torch.FloatTensor)
        state_embed = self.basenet(state)
        ap = self.action_head(state_embed)
        v = self.value_head(state_embed)
        return ap, v


class ConvPolicy(nn.Module):
    def __init__(self, env: gym.Env):
        super().__init__()

        img_height, img_width, img_channels = env.observation_space.shape

        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1, stride=1),
            nn.ReLU()
        )

        o = self.conv(torch.zeros(1, img_channels, img_height, img_width))
        conv_out_size = int(np.prod(o.size()))

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        state = torch.from_numpy(state).float() / 256
        if len(state.shape) == 3:
            state = state.view(1, *state.shape)
            state = state.permute(0, 3, 1, 2)
        elif len(state.shape) == 2:
            state = state.view(1, 1, *state.shape)

        conv_out = self.conv(state).view(state.size(0), -1)
        ap = self.fc(conv_out)
        return ap
