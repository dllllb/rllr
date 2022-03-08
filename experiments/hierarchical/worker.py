import torch
import torch.nn as nn
from torch.nn import functional as F

from rllr.models.ppo import FixedCategorical


def init_params(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias, 0)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        out_shape = self.conv(torch.zeros(1, *obs_shape)).shape
        conv_out = out_shape[1] * out_shape[2] * out_shape[3]

        self.fc = nn.Sequential(
            nn.Linear(in_features=conv_out, out_features=1024),
            nn.ReLU(inplace=True),
        )

        self.output_size = 1024

    def forward(self, t):
        hid = t.float() / 255.
        hid = self.conv(hid).reshape(t.shape[0], -1)
        hid = self.fc(hid)
        return hid


class WorkerPolicyModel(nn.Module):

    def __init__(self, state_shape, n_actions, goal_size):
        super(WorkerPolicyModel, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.is_recurrent = False

        self.enc = Encoder(self.state_shape)

        self.policy = nn.Sequential(
            nn.Linear(in_features=self.enc.output_size + goal_size, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=self.n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(in_features=self.enc.output_size + goal_size, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=1)
        )

        self.apply(init_params)

    def forward(self, inputs, rnn_hxs, masks):
        enc = self.enc(inputs['image'])
        enc = torch.cat([enc, inputs['goal']], dim=1)
        value = self.value(enc)
        logits = self.policy(enc)
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
