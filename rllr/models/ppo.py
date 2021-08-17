import torch
from torch import nn as nn
from torch.nn import functional as F


class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


class ActorNetwork(nn.Module):
    """
    Actor is a policy network. Given state it evaluates
    probability of action given state or sample an action
    """

    def __init__(self, action_size, state_encoder, hidden_size):
        super().__init__()
        self.state_encoder = state_encoder
        input_size = state_encoder.output_size
        fc_layers = []
        if type(hidden_size) == int:
            fc_layers += [
                nn.Linear(input_size, hidden_size),
                nn.ReLU(inplace=False),
                nn.Linear(hidden_size, action_size)
            ]
        elif type(hidden_size) == list:
            for hs in hidden_size:
                fc_layers.append(nn.Linear(input_size, hs))
                fc_layers.append(nn.ReLU(inplace=False))
                input_size = hs
            fc_layers.append(nn.Linear(input_size, action_size))
        else:
            AttributeError(f"unknown type of {hidden_size} parameter")

        self.fc = nn.Sequential(*fc_layers)
        self.logstd = nn.Parameter(torch.zeros((action_size,)))
        self.output_size = action_size

    def forward(self, states):
        states_encoding = self.state_encoder(states)
        mu = self.fc(states_encoding)
        std = self.logstd.exp()
        return FixedNormal(mu, std)


class CriticNetwork(nn.Module):
    """
    Critic network estimates value function
    """

    def __init__(self, state_encoder, hidden_size):
        super(CriticNetwork, self).__init__()
        self.state_encoder = state_encoder
        input_size = state_encoder.output_size
        fc_layers = []
        if type(hidden_size) == int:
            fc_layers += [
                nn.Linear(input_size, hidden_size),
                nn.ReLU(inplace=False),
                nn.Linear(hidden_size, 1)
            ]
        elif type(hidden_size) == list:
            for hs in hidden_size:
                fc_layers.append(nn.Linear(input_size, hs))
                fc_layers.append(nn.ReLU(inplace=False))
                input_size = hs
            fc_layers.append(nn.Linear(input_size, 1))
        else:
            AttributeError(f"unknown type of {hidden_size} parameter")
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, states):
        x = self.state_encoder(states)
        return self.fc(x)


class ActorCriticNetwork(nn.Module):
    """
    Actor-critic model
    """

    def __init__(self, action_size, actor_state_encoder, critic_state_encoder, actor_hidden_size, critic_hidden_size):
        super(ActorCriticNetwork, self).__init__()
        self.actor = ActorNetwork(action_size, actor_state_encoder, actor_hidden_size)
        self.critic = CriticNetwork(critic_state_encoder, critic_hidden_size)

    def act(self, states, deterministic=False):
        dist = self.actor.forward(states)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        return self.critic.forward(states), action, dist.log_probs(action)

    def get_value(self, states):
        return self.critic.forward(states)

    def evaluate_actions(self, states, actions):
        dist = self.actor.forward(states)
        values = self.critic.forward(states)
        return values, dist.log_probs(actions), dist.entropy().mean()
