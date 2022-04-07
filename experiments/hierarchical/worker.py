import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import init_params

from rllr.models.ppo import FixedCategorical


class WorkerPolicyModel(nn.Module):
    def __init__(self, state_size, n_actions):
        super(WorkerPolicyModel, self).__init__()
        self.n_actions = n_actions
        self.is_recurrent = False

        self.policy = nn.Sequential(
            nn.Linear(in_features=state_size, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=self.n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(in_features=state_size, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=1)
        )

        self.apply(init_params)

    def forward(self, inputs, rnn_hxs, masks):
        value = self.value(inputs)
        logits = self.policy(inputs)
        dist = FixedCategorical(logits=F.log_softmax(logits, dim=1))
        return dist, value, rnn_hxs

    def act(self, states, rnn_hxs, masks, deterministic=False):
        dist, value, rnn_hxs = self.forward(states, rnn_hxs, masks)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        return value, action, dist.log_probs(action), rnn_hxs

    def get_value(self, states, rnn_hxs, masks):
        dist, value, rnn_hxs = self.forward(states, rnn_hxs, masks)
        return value

    def evaluate_actions(self, states, actions, rnn_hxs, masks):
        dist, value, rnn_hxs = self.forward(states, rnn_hxs, masks)
        return value, dist.log_probs(actions), dist.entropy().mean(), rnn_hxs
