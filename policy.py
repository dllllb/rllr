from torch.distributions import Categorical

from learner import Learner, Updater, BufferedLearner
import torch.nn as nn
import torch
from constants import *


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


class NNPolicy(BufferedLearner, Policy):
    def __init__(self, model: nn.Module, updater: Updater):
        super().__init__(updater)
        self.model = model

    def __call__(self, state):
        take_probs = self.model(state)

        is_nan(take_probs, 'probs')
        
        #c = Categorical(take_probs) # with softmax activation
        c = Categorical(logits=take_probs)
        action = c.sample()

        log_p = c.log_prob(action).view(1)
        if is_nan(log_p, 'log_p'):
            print(f'lop_prob is Nan for take_probs {take_probs}, action {action.item()}')

        return action.item(), log_p


class NNPolicyAV(BufferedLearner, Policy):
    def __init__(self, model: nn.Module, updater: Updater):
        super().__init__(updater)
        self.model = model

    def __call__(self, state):
        ap, v = self.model(state)
        c = Categorical(ap)
        action = c.sample()
        return action.item(), (c.log_prob(action).view(1), v.item())
