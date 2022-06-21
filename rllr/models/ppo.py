import gym
import torch
from torch import nn as nn
from torch.nn import functional as F

from .encoders import OneHot


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


def make_action_encodings(n_actions, hidden_size, output_size):
    input_size = n_actions
    fc_layers = [OneHot(n_actions)]
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


class TensorDict(dict):
    def __getitem__(self, item):
        if type(item) == int:
            return TensorDict({k: v[item] for k, v in self.items()})
        else:
            return super().__getitem__(item)

    def __iter__(self):
        maxind = list(self.values())[0].shape[0]
        for ind in range(maxind):
            yield {k: v[ind] for k, v in self.items()}

    def cpu(self):
        return TensorDict({k: v.cpu() for k, v in self.items()})

    def numpy(self):
        return TensorDict({k: v.numpy() for k, v in self.items()})

    def mean(self, mean_by_key=False):
        means = TensorDict({k: v.mean() for k, v in self.items()})
        if mean_by_key:
            return means
        else:
            return torch.stack([v for v in means.values()]).mean()


class MultiDistribution(dict):
    def log_probs(self, action):
        return {key: dist.log_prob(action[key][:, 0]).unsqueeze(-1) for key, dist in self.items()}

    def mode(self):
        return TensorDict({key: dist.mode() for key, dist in self.items()})

    def entropy(self):
        return TensorDict({key: dist.entropy() for key, dist in self.items()})

    def sample(self):
        return TensorDict({key: dist.sample() for key, dist in self.items()})


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

    def forward(self, states, rnn_hxs, masks, encodings_feeded=False, deterministic=False, action_mask=None):
        if encodings_feeded:
            states_encoding = states
        else:
            if self.is_recurrent:
                states_encoding, rnn_hxs = self.state_encoder(states, rnn_hxs, masks)
            else:
                states_encoding = self.state_encoder(states)

        logits = self.logits(states_encoding)
        if action_mask is not None:
            logits = torch.where(action_mask == 0, torch.zeros_like(logits)-float('inf'), logits)

        dist = FixedCategorical(logits=F.log_softmax(logits, dim=1))

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        return action, dist, rnn_hxs


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

    def forward(self, states, rnn_hxs, masks, encodings_feeded=False, deterministic=False, action_mask=None):
        if encodings_feeded:
            states_encoding = states
        else:
            if self.is_recurrent:
                states_encoding, rnn_hxs = self.state_encoder(states, rnn_hxs, masks)
            else:
                states_encoding = self.state_encoder(states)

        mu = self.fc(states_encoding)
        std = self.logstd.exp()
        dist = FixedNormal(mu, std)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        return action, dist, rnn_hxs


class MultiOutputActorNetwork(nn.Module):
    def __init__(self, action_space, state_encoder, hidden_size, is_recurrent):
        super().__init__()
        self.state_encoder = state_encoder
        input_size = state_encoder.output_size

        self.fc = nn.ModuleDict()
        self.logstd = nn.ParameterDict()
        self.output_size = dict()
        self.spaces = action_space

        for key in action_space:

            if key == 'action_mask':
                continue

            space = action_space[key]
            if type(space) == gym.spaces.Box:
                action_size = space.shape[0]
                self.fc[key] = make_mlp(input_size, hidden_size, action_size)
                self.logstd[key] = nn.Parameter(torch.zeros((action_size,)))
                self.output_size[key] = action_size
            elif type(space) == gym.spaces.Discrete:
                action_size = space.n
                self.fc[key] = make_mlp(input_size, hidden_size, action_size)
                self.output_size[key] = action_size
            else:
                raise NotImplementedError(f'{space} not supported')
        self.is_recurrent = is_recurrent

    def forward(self, states, rnn_hxs, masks, encodings_feeded=False, deterministic=False, action_mask=None):
        if encodings_feeded:
            states_encoding = states
        else:
            if self.is_recurrent:
                states_encoding, rnn_hxs = self.state_encoder(states, rnn_hxs, masks)
            else:
                states_encoding = self.state_encoder(states)

        distributions = MultiDistribution()
        for space_name in self.spaces:
            space = self.spaces[space_name]
            if type(space) == gym.spaces.Box:
                mu = self.fc[space_name](states_encoding)
                std = self.logstd[space_name].exp()
                distributions[space_name] = FixedNormal(mu, std)
            elif type(space) == gym.spaces.Discrete:
                logits = self.fc[space_name](states_encoding)
                distributions[space_name] = FixedCategorical(logits=F.log_softmax(logits, dim=1))
            else:
                raise NotImplementedError(f'{space} not supported')

        if deterministic:
            action = distributions.mode()
        else:
            action = distributions.sample()

        return action, distributions, rnn_hxs


class ConditionalMultiOutputActorNetwork(nn.Module):
    def __init__(self, action_space, state_encoder, hidden_size, is_recurrent, heads_order):
        super().__init__()
        self.state_encoder = state_encoder
        input_size = state_encoder.output_size

        self.fc = nn.ModuleDict()
        self.logstd = nn.ParameterDict()
        self.output_size = dict()
        self.spaces = action_space
        self.action_encodings = nn.ModuleDict()
        self.heads_order = heads_order

        cumulative_input_size = 0
        for key in heads_order:

            if key == 'action_mask':
                continue

            space = action_space[key]
            if type(space) == gym.spaces.Box:
                action_size = space.shape[0]
                self.fc[key] = make_mlp(input_size + cumulative_input_size, hidden_size, action_size)
                self.logstd[key] = nn.Parameter(torch.zeros((action_size,)))
                self.output_size[key] = action_size
                self.action_encodings[key] = make_mlp(action_size, hidden_size, input_size)
            elif type(space) == gym.spaces.Discrete:
                action_size = space.n
                self.fc[key] = make_mlp(input_size + cumulative_input_size, hidden_size, action_size)
                self.output_size[key] = action_size
                self.action_encodings[key] = make_action_encodings(action_size, hidden_size, input_size)
            else:
                raise NotImplementedError(f'{space} not supported')
            cumulative_input_size += input_size
        self.is_recurrent = is_recurrent

    def forward(self, states, rnn_hxs, masks, encodings_feeded=False, deterministic=False, action_mask=None):
        if encodings_feeded:
            states_encoding = states
        else:
            if self.is_recurrent:
                states_encoding, rnn_hxs = self.state_encoder(states, rnn_hxs, masks)
            else:
                states_encoding = self.state_encoder(states)

        distributions = MultiDistribution()
        actions = dict()
        encoded_actions = list()
        for space_name in self.heads_order:
            space = self.spaces[space_name]
            state_input = torch.cat(encoded_actions + [states_encoding], dim=-1)

            if type(space) == gym.spaces.Box:
                mu = self.fc[space_name](state_input)
                std = self.logstd[space_name].exp()
                distributions[space_name] = FixedNormal(mu, std)

            elif type(space) == gym.spaces.Discrete:
                logits = self.fc[space_name](state_input)
                if action_mask is not None:
                    if action_mask.dim() > 2:
                        mask = action_mask.sum(dim=[i for i in range(2, action_mask.dim())])
                    else:
                        mask = action_mask
                    logits = torch.where(mask == 0, torch.zeros_like(logits)-float('inf'), logits)
                distributions[space_name] = FixedCategorical(logits=F.log_softmax(logits, dim=1))

            else:
                raise NotImplementedError(f'{space} not supported')

            if deterministic:
                actions[space_name] = distributions[space_name].mode()
            else:
                actions[space_name] = distributions[space_name].sample()

            encoded_actions.append(self.action_encodings[space_name](actions[space_name].squeeze(-1)))
            if action_mask is not None:
                new_mask = list()
                for m, a in zip(action_mask, actions[space_name][:, 0]):
                    new_mask.append(m[a])
                action_mask = torch.stack(new_mask, dim=0)

        actions = TensorDict(actions)
        return actions, distributions, rnn_hxs


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

    def forward(self, states, rnn_hxs, masks, encodings_feeded=False):
        if encodings_feeded:
            states_encoding = states
        else:
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

    def forward(self, states, rnn_hxs, masks, encodings_feeded=False):
        if encodings_feeded:
            states_encoding = states
        else:
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
                 use_intrinsic_motivation=False, is_recurrent=False, same_encoders=False, heads_order=None):
        super(ActorCriticNetwork, self).__init__()
        if type(action_space) == gym.spaces.Box:
            self.actor = ContiniousActorNetwork(action_space.shape[0], actor_state_encoder, actor_hidden_size,
                                                is_recurrent)
        elif type(action_space) == gym.spaces.Discrete:
            self.actor = DiscreteActorNetwork(action_space.n, actor_state_encoder, actor_hidden_size, is_recurrent)
        elif type(action_space) == gym.spaces.Dict:
            if heads_order is not None:
                self.actor = ConditionalMultiOutputActorNetwork(action_space,
                                                                actor_state_encoder,
                                                                actor_hidden_size,
                                                                is_recurrent,
                                                                heads_order=heads_order)
            else:
                self.actor = MultiOutputActorNetwork(action_space, actor_state_encoder, actor_hidden_size, is_recurrent)
        else:
            raise NotImplementedError(f'{action_space} not supported')

        if use_intrinsic_motivation:
            self.critic = IMCriticNetwork(critic_state_encoder, critic_hidden_size, is_recurrent)
        else:
            self.critic = CriticNetwork(critic_state_encoder, critic_hidden_size, is_recurrent)

        self.is_recurrent = is_recurrent
        self.same_encoders = same_encoders
        self.state_encoder = actor_state_encoder

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
        action_mask = None
        if isinstance(states, dict):
            if 'action_mask' in states:
                action_mask = states['action_mask']

        if getattr(self, 'same_encoders', False):
            if self.is_recurrent:
                states, rnn_hxs = self.state_encoder(states, rnn_hxs, masks)
            else:
                states = self.state_encoder(states)

        action, dist, actor_rnn_hxs = self.actor.forward(states, rnn_hxs, masks,
                                                         encodings_feeded=getattr(self, 'same_encoders', False),
                                                         deterministic=deterministic,
                                                         action_mask=action_mask)

        return self.critic.forward(states, rnn_hxs, masks, encodings_feeded=getattr(self, 'same_encoders', False)), \
               action, dist.log_probs(action), actor_rnn_hxs

    def get_value(self, states, rnn_hxs, masks):
        return self.critic.forward(states, rnn_hxs, masks)

    def evaluate_actions(self, states, actions, rnn_hxs, masks):
        if getattr(self, 'same_encoders', False):
            if self.is_recurrent:
                states, rnn_hxs = self.state_encoder(states, rnn_hxs, masks)
            else:
                states = self.state_encoder(states)

        action_mask = None
        if isinstance(states, dict):
            if 'action_mask' in states:
                action_mask = states['action_mask']

        _, dist, rnn_hxs = self.actor.forward(states, rnn_hxs, masks,
                                              encodings_feeded=getattr(self, 'same_encoders', False),
                                              action_mask=action_mask)
        values = self.critic.forward(states, rnn_hxs, masks,
                                     encodings_feeded=getattr(self, 'same_encoders', False))
        return values, dist.log_probs(actions), dist.entropy().mean(), rnn_hxs

    def parameters(self, recurse: bool = True):
        for name, param in self.named_parameters(recurse=recurse):
            if 'embed_layers' not in name:
                yield param

    def embed_parameters(self, recurse: bool = True):
        for name, param in self.named_parameters(recurse=recurse):
            if 'embed_layers' in name:
                #param.requires_grad = False
                yield param
