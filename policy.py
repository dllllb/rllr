from torch.distributions import Categorical

from learner import Learner, Updater, BufferedLearner
import torch.nn as nn
import torch
from constants import *
import random
import math


class Policy(Learner):
    def __call__(self, state):
        """
        :param state: current state of the environment
        :return: proposed action
        """
        pass


class RandomActionPolicy(Policy):
    def __init__(self, env):
        self.actions = env.action_space

    def __call__(self, _):
        return self.actions.sample(), None

class CategoricalActionPolicy(Policy):
    def __init__(self, sampler):
        self.sampler = sampler

    def __call__(self, _):
        return self.sampler.sample().item(), None

class NNPolicy(BufferedLearner, Policy):
    def __init__(self, model: nn.Module, updater: Updater):
        super().__init__(updater)
        self.model = model

    def __call__(self, state):
        take_probs = self.model(state)
        c = Categorical(logits=take_probs)
        action = c.sample()
        return action.item(), c.log_prob(action).view(1)

class NNExplorationPolicy(BufferedLearner, Policy):
    def __init__(self, model: nn.Module, updater: Updater):
        super().__init__(updater)
        self.model = model

        self.epochs = 1000
        self.steps_per_epoch = 1000.0
        self.step = 0

    def __call__(self, state):
        take_probs = self.model(state)
        v = None
        if isinstance(take_probs, tuple):
            take_probs, v = take_probs[0], take_probs[1]
        c = Categorical(logits=take_probs)

        assert not torch.isnan(take_probs).any(), f'\nnan in log_prob {take_probs}\n++++++++++++++++'
        #assert not torch.isnan(v), f'\nnan in value {v}\n+++++++++++++++++++'

        #temperature = math.exp(-1*self.step/(self.steps_per_epoch*self.epochs)) # exponential decay
        #temperature = abs(math.sin(self.step/self.steps_per_epoch)) # oscilation
        temperature = -1

        if random.random() > 0.1 * temperature:
            action = c.sample()
        else:
            action = torch.randint(0, take_probs.size(-1), (1,)).to(take_probs.device)

        self.step += 1
        if v is None:
            return action.item(), (c.log_prob(action).view(1), c.entropy())
        else:
            return action.item(), (c.log_prob(action).view(1), c.entropy(), v)


class NNPolicyAV(BufferedLearner, Policy):
    def __init__(self, model: nn.Module, updater: Updater):
        super().__init__(updater)
        self.model = model

    def __call__(self, state):
        ap, v = self.model(state)
        c = Categorical(ap)
        action = c.sample()
        return action.item(), (c.log_prob(action).view(1), v.item())
