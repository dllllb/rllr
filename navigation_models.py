import gym
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from constants import *
import cv2


def prepare_state(state):
    if isinstance(state, np.ndarray):
        state = torch.from_numpy(state).float() / 256
    if len(state.shape) == 3:
        state = state.view(1, *state.shape)
        state = state.permute(0, 3, 1, 2)
    elif len(state.shape) == 2:
        state = state.view(1, 1, *state.shape)
    return state.to(DEVICE)

class ConvBody(nn.Module):

    def __init__(self, env):
        super().__init__()
        img_height, img_width, img_channels = env.observation_space.shape

        self.convs = nn.ModuleList([
            nn.Conv2d(img_channels, 32, kernel_size=8, stride=4),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.Conv2d(64, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 1, kernel_size=1, stride=1),
        ])

        #HooksF = [Hook(conv, f'conv_{i+1}') for i, conv in enumerate(self.convs)]
        #HooksB = [Hook(conv, f'conv_{i+1}', backward=True) for i, conv in enumerate(self.convs)]

        with torch.no_grad():
            self.norms = []
            x = torch.randn(1, img_channels, img_height, img_width)
            for conv in self.convs:
                x = conv(x)
                self.norms.append(nn.LayerNorm(normalized_shape=x.size()[1:]))
            self.norm = nn.ModuleList(self.norms)

    def forward(self, x):
        for i, conv, norm in zip(range(len(self.convs)), self.convs, self.norms):
            x = F.relu(norm(conv(x)))
        return x


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

        self.conv = ConvBody(env) #state_embed_block(img_channels)

        self.conv_target = ConvBody(env) #state_embed_block(img_channels)

        o = self.conv(torch.zeros(1, img_channels, img_height, img_width))
        conv_out_size = int(np.prod(o.size()))

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size*2, 128),
            nn.LayerNorm([128]),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n),
            nn.LogSoftmax(dim=-1)
        )
        #HooksF = [Hook(layer, f'head_{layer.__repr__()}') for layer in self.fc]
        #HooksB = [Hook(layer, f'head_{layer.__repr__()}', backward=True) for layer in self.fc]

        

    def forward(self, state):
        current_state_, desired_state_ = state
        current_state = prepare_state(current_state_['image'])
        desired_state = prepare_state(desired_state_['image'])
        #current_state = (current_state - current_state.min())/(current_state.max() - current_state.min())
        #desired_state = (desired_state - desired_state.mean())/(desired_state.max() - desired_state.min())

        conv_out = self.conv(current_state).view(current_state.size(0), -1)
        conv_target_out = self.conv_target(desired_state).view(desired_state.size(0), -1)
        h = torch.cat((conv_out, conv_target_out), dim=1)
        #h = conv_target_out - conv_out

        ap = self.fc(h)
        return ap

class StateAPINavPolicy(nn.Module):
    def __init__(self, env: gym.Env):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(3, 64),
            #nn.LayerNorm([64]),
            nn.ReLU(),
            nn.Linear(64, 3),#env.action_space.n),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, state):
        current_state, desired_state = state
        current_pos = torch.as_tensor(current_state['agent_pos']).float().to(DEVICE)
        desired_pos = torch.as_tensor(desired_state['agent_pos']).float().to(DEVICE)
        current_dir = torch.as_tensor(current_state['direction']).float().to(DEVICE)

        #h = torch.cat((current_state, desired_state), dim=0)
        h = torch.cat((desired_pos - current_pos, current_dir), dim=0)

        ap = self.fc(h)
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
