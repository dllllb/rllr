import gym
import torch
import numpy as np
from torch import nn
from constants import *


def prepare_state(state):
    if isinstance(state, np.ndarray):
        state = torch.from_numpy(state).float() / 256
    if len(state.shape) == 3:
        state = state.view(1, *state.shape)
        state = state.permute(0, 3, 1, 2)
    elif len(state.shape) == 2:
        state = state.view(1, 1, *state.shape)
    return state.to(DEVICE)


def state_embed_block(img_channels):
    return nn.Sequential(
        nn.Conv2d(img_channels, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=2, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 1, kernel_size=1, stride=1),
        nn.ReLU()
    )


class ConvNavPolicy(nn.Module):
    def __init__(self, env: gym.Env):
        super().__init__()

        img_height, img_width, img_channels = env.observation_space.shape

        self.conv = state_embed_block(img_channels)

        self.conv_target = state_embed_block(img_channels)

        o = self.conv(torch.zeros(1, img_channels, img_height, img_width))
        conv_out_size = int(np.prod(o.size()))

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size*1, 128),
            nn.LayerNorm([128]),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n),
            #nn.Softmax(dim=-1)
        )
        self.act = nn.LogSoftmax(dim=-1)

    def forward(self, state):
        current_state, desired_state = state
        current_state = prepare_state(current_state)
        desired_state = prepare_state(desired_state)
        current_state = (current_state - current_state.min())/(current_state.max() - current_state.min())
        desired_state = (desired_state - desired_state.mean())/(desired_state.max() - desired_state.min())

        if torch.isnan(current_state).any().item():
            print(f'+++++++++++++++++nan in current_state values +++++++++++++++++++\n')

        if torch.isinf(current_state).any().item():
            print(f'+++++++++++++++++infinity in current_state  values +++++++++++++++++++\n')

        if torch.isnan(desired_state).any().item():
            print(f'+++++++++++++++++nan in desired_state values +++++++++++++++++++\n')

        if torch.isinf(desired_state).any().item():
            print(f'+++++++++++++++++infinity in desired_state  values +++++++++++++++++++\n')

        conv_out = self.conv(current_state).view(current_state.size(0), -1)
        conv_target_out = self.conv_target(desired_state).view(desired_state.size(0), -1)
        #h = torch.cat((conv_out, conv_target_out), dim=1)
        h = conv_target_out - conv_out

        if torch.isnan(h).any().item():
            print(f'+++++++++++++++++nan in conv layer values +++++++++++++++++++\n')

        if torch.isinf(h).any().item():
            print(f'+++++++++++++++++infinity in conv layer  values +++++++++++++++++++\n')

        ap = self.fc(h)

        if torch.isnan(ap).any().item():
            print(f'+++++++++++++++++nan in linear layer values +++++++++++++++++++\n')

        if torch.isinf(ap).any().item():
            print(f'+++++++++++++++++infinity in linear layer  values +++++++++++++++++++\n')


        ap = self.act(ap)
        return ap

class GoalConvNavPolicy(nn.Module):
    def __init__(self, model: nn.Module, env: gym.Env, starter: torch.Tensor=None):
        super().__init__()

        self.model = model

        if starter is None:
            img_height, img_width, img_channels = env.observation_space.shape
            self.starter = nn.Parameter(torch.randn(img_height, img_width, img_channels))
        else:
            starter = nn.Parameter(starter)

    def forward(self, state):
        input = (state, self.starter)
        return self.model(input)
