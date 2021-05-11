import torch
from torch import nn as nn


class ActorNetwork(nn.Module):

    """
    Actor network. Input is current state and output is action in continuous action space.
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

        fc_layers.append(nn.Tanh())

        self.fc = nn.Sequential(*fc_layers)
        self.output_size = action_size

    def forward(self, states):
        states_encoding = self.state_encoder(states)
        return self.fc(states_encoding)


class CriticNetwork(nn.Module):
    """
    Critic network receives actions and states and returns value for of action in given state
    """

    def __init__(self, action_size, state_encoder, hidden_size):
        super(CriticNetwork, self).__init__()
        self.state_encoder = state_encoder
        input_size = state_encoder.output_size + action_size
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

    def forward(self, states, goal_embs):
        x = self.state_encoder(states)
        x = torch.cat((x, goal_embs), 1)
        return self.fc(x)


class ActorCriticNetwork(nn.Module):
    """
    Actor-critic model for DDPG
    """

    def __init__(self, action_size, actor_state_encoder, critic_state_encoder, actor_hidden_size, critic_hidden_size):
        super(ActorCriticNetwork, self).__init__()
        self.actor = ActorNetwork(action_size, actor_state_encoder, actor_hidden_size)
        self.critic = CriticNetwork(action_size, critic_state_encoder, actor_hidden_size)
        self.target_actor = ActorNetwork(action_size, actor_state_encoder, actor_hidden_size)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic = CriticNetwork(action_size, critic_state_encoder, critic_hidden_size)
        self.target_critic.load_state_dict(self.critic.state_dict())