import torch
from torch import nn as nn
from torch.distributions import Categorical
from torch.nn import functional as F


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

        fc_layers.append(nn.Softmax(dim=1))
        self.fc = nn.Sequential(*fc_layers)
        self.output_size = action_size

    def forward(self, states, actions=None):
        states_encoding = self.state_encoder(states)
        probs = self.fc(states_encoding)
        action_dist = Categorical(probs)
        actions = actions if actions is not None else action_dist.sample()
        action_log_prob = action_dist.log_prob(actions.squeeze())
        entropy = action_dist.entropy()
        return actions.detach(), action_log_prob.unsqueeze(1), entropy


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


class ActorCritic(nn.Module):
    """
    Actor-critic model
    """

    def __init__(self, action_size, actor_state_encoder, critic_state_encoder, actor_hidden_size, critic_hidden_size):
        super(ActorCritic, self).__init__()
        self.actor = ActorNetwork(action_size, actor_state_encoder, actor_hidden_size)
        self.critic = CriticNetwork(critic_state_encoder, critic_hidden_size)

    def act(self, states):
        actions, _, _ = self.actor.forward(states, None)
        return actions

    def evaluate(self, states, actions):
        _, logprobs, entropy = self.actor.forward(states, actions)
        values = self.critic.forward(states)
        return values, logprobs, entropy.reshape(-1, 1)