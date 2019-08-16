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
        c = Categorical(self.model(state))
        return c


class ConvPolicy(nn.Module):
    def __init__(self, env: gym.Env):
        super().__init__()

        img_height, img_width, img_channels = env.observation_space.shape

        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        o = self.conv(torch.zeros(1, img_channels, img_height, img_width))
        conv_out_size = int(np.prod(o.size()))

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n),
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
        c = Categorical(self.fc(conv_out))
        return c