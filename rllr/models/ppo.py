import torch
from torch import nn as nn
from torch.distributions import Categorical
from torch.nn import functional as F


class ActorNetwork(nn.Module):
    """
    Actor is a policy network. Given state it evaluates
    probability of action given state or sample an action
    """

    def __init__(self, state_size, action_size, seed=42, hidden_size1=64, hidden_size2=64):
        super(ActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layer1 = nn.Linear(state_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, action_size)

    def forward(self, states, actions=None):
        x = F.relu(self.layer1(states))
        x = F.relu(self.layer2(x))
        probs = F.softmax(self.layer3(x), dim=1)

        action_dist = Categorical(probs)
        actions = actions if actions is not None else action_dist.sample()

        action_log_prob = action_dist.log_prob(actions.squeeze())
        entropy = action_dist.entropy()

        return actions.detach(), action_log_prob.unsqueeze(1), entropy


class CriticNetwork(nn.Module):
    """
    Critic network estimates value function
    """

    def __init__(self, state_size, seed=42, hidden_size1=64, hidden_size2=64):
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layer1 = nn.Linear(state_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, 1)

    def forward(self, states):
        x = F.relu(self.layer1(states))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class ActorCritic(nn.Module):
    """
    Actor-critic model
    """

    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.actor = ActorNetwork(state_size, action_size)
        self.critic = CriticNetwork(state_size)

    def act(self, states):
        actions, _, _ = self.actor.forward(states, None)
        return actions

    def evaluate(self, states, actions):
        _, logprobs, entropy = self.actor.forward(states, actions)

        values = self.critic.forward(states)
        return values, logprobs, entropy.reshape(-1, 1)