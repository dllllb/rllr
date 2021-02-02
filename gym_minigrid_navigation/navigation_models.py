import torch

from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    """
    Simple multilayer network for DQN agent
    """

    def __init__(self, state_size, action_size, seed, hidden_size1=64, hidden_size2=64):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.layer_m1 = nn.Linear(state_size, hidden_size1)
        self.layer_m2 = nn.Linear(hidden_size1, hidden_size2)

        self.layer_w1 = nn.Linear(state_size, hidden_size1)
        self.layer_w2 = nn.Linear(hidden_size1, hidden_size2)

        self.layer3 = nn.Linear(2 * hidden_size2, hidden_size2)
        self.layer4 = nn.Linear(hidden_size2, action_size)

    def forward(self, state, goal_state):
        x_m = F.relu(self.layer_m1(goal_state))
        x_m = F.relu(self.layer_m2(x_m))

        x_w = F.relu(self.layer_w1(state))
        x_w = F.relu(self.layer_w2(x_w))

        x = torch.cat((x_m, x_w), 1)
        x = F.relu(self.layer3(x))
        return self.layer4(x)


class Flatter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, state, goal_state):
        state = state[:, 1:-1, 1:-1].flatten(start_dim=1)
        goal_state = goal_state[:, 1:-1, 1:-1].flatten(start_dim=1)
        return self.model(state, goal_state)
