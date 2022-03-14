import gym
import torch
from torch import nn as nn
from torch.nn import functional as F


def make_mlp(input_size, hidden_size, output_size):
    fc_layers = []
    if type(hidden_size) == int:
        fc_layers += [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_size, output_size)
        ]
    elif type(hidden_size) == list:
        for hs in hidden_size:
            fc_layers.append(nn.Linear(input_size, hs))
            fc_layers.append(nn.ReLU(inplace=False))
            input_size = hs
        fc_layers.append(nn.Linear(input_size, output_size))
    else:
        AttributeError(f"unknown type of {hidden_size} parameter")

    return nn.Sequential(*fc_layers)


class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


class DiscreteActorNetwork(nn.Module):
    """
    Actor is a policy network. Given state it evaluates
    probability of action given state or sample an action
    """

    def __init__(self, action_size, state_encoder, hidden_size, is_recurrent):
        super().__init__()
        self.state_encoder = state_encoder
        input_size = state_encoder.output_size
        self.logits = make_mlp(input_size, hidden_size, action_size)
        self.output_size = action_size
        self.is_recurrent = is_recurrent

    def forward(self, states, rnn_hxs, masks):
        if self.is_recurrent:
            states_encoding, rnn_hxs = self.state_encoder(states, rnn_hxs, masks)
        else:
            states_encoding = self.state_encoder(states)

        logits = self.logits(states_encoding)
        return FixedCategorical(logits=F.log_softmax(logits, dim=1)), rnn_hxs


class ContiniousActorNetwork(nn.Module):
    """
    Actor is a policy network. Given state it evaluates
    probability of action given state or sample an action
    """

    def __init__(self, action_size, state_encoder, hidden_size, is_recurrent):
        super().__init__()
        self.state_encoder = state_encoder
        input_size = state_encoder.output_size
        self.fc = make_mlp(input_size, hidden_size, action_size)
        self.logstd = nn.Parameter(torch.zeros((action_size,)))
        self.output_size = action_size
        self.is_recurrent = is_recurrent

    def forward(self, states, rnn_hxs, masks):
        if self.is_recurrent:
            states_encoding, rnn_hxs = self.state_encoder(states, rnn_hxs, masks)
        else:
            states_encoding = self.state_encoder(states)

        mu = self.fc(states_encoding)
        std = self.logstd.exp()
        return FixedNormal(mu, std), rnn_hxs


class CriticNetwork(nn.Module):
    """
    Critic network estimates value function
    """

    def __init__(self, state_encoder, hidden_size, is_recurrent):
        super(CriticNetwork, self).__init__()
        self.state_encoder = state_encoder
        input_size = state_encoder.output_size
        self.fc = make_mlp(input_size, hidden_size, 1)
        self.is_recurrent = is_recurrent

    def forward(self, states, rnn_hxs, masks):
        if self.is_recurrent:
            states_encoding, rnn_hxs = self.state_encoder(states, rnn_hxs, masks)
        else:
            states_encoding = self.state_encoder(states)

        return self.fc(states_encoding)


class IMCriticNetwork(nn.Module):
    """
    Critic network estimates external and internal expected returns
    """

    def __init__(self, state_encoder, hidden_size, is_recurrent):
        super(IMCriticNetwork, self).__init__()
        self.state_encoder = state_encoder
        input_size = state_encoder.output_size
        self.ext_head = make_mlp(input_size, hidden_size, 1)
        self.int_head = make_mlp(input_size, hidden_size, 1)

        self.is_recurrent = is_recurrent

    def forward(self, states, rnn_hxs, masks):
        if self.is_recurrent:
            states_encoding, rnn_hxs = self.state_encoder(states, rnn_hxs, masks)
        else:
            states_encoding = self.state_encoder(states)

        return self.ext_head(states_encoding), self.int_head(states_encoding)


class ActorCriticNetwork(nn.Module):
    """
    Actor-critic model
    """

    def __init__(self, action_space, actor_state_encoder, critic_state_encoder, actor_hidden_size, critic_hidden_size,
                 use_intrinsic_motivation=False, is_recurrent=False):
        super(ActorCriticNetwork, self).__init__()
        if type(action_space) == gym.spaces.Box:
            self.actor = ContiniousActorNetwork(action_space.shape[0], actor_state_encoder, actor_hidden_size, is_recurrent)
        elif type(action_space) == gym.spaces.Discrete:
            self.actor = DiscreteActorNetwork(action_space.n, actor_state_encoder, actor_hidden_size, is_recurrent)
        else:
            raise NotImplementedError(f'{action_space} not supported')

        if use_intrinsic_motivation:
            self.critic = IMCriticNetwork(critic_state_encoder, critic_hidden_size, is_recurrent)
        else:
            self.critic = CriticNetwork(critic_state_encoder, critic_hidden_size, is_recurrent)

        self.is_recurrent = is_recurrent

        def init_params(m):
            classname = m.__class__.__name__
            if classname.find("Linear") != -1:
                m.weight.data.normal_(0, 1)
                m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if classname.find("LSTM") != -1:
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0)
                    elif 'weight' in name:
                        nn.init.orthogonal_(param)
            if classname.find('Conv2d') != -1:
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0)

        self.apply(init_params)

    def act(self, states, rnn_hxs, masks, deterministic=False):
        dist, actor_rnn_hxs = self.actor.forward(states, rnn_hxs, masks)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        return self.critic.forward(states, rnn_hxs, masks), action, dist.log_probs(action), actor_rnn_hxs

    def get_value(self, states, rnn_hxs, masks):
        return self.critic.forward(states, rnn_hxs, masks)

    def evaluate_actions(self, states, actions, rnn_hxs, masks):
        dist, rnn_hxs = self.actor.forward(states, rnn_hxs, masks)
        values = self.critic.forward(states, rnn_hxs, masks)
        return values, dist.log_probs(actions), dist.entropy().mean(), rnn_hxs
