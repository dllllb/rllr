from torch.distributions import Categorical

from learner import Learner, Updater, BufferedLearner
import torch.nn as nn


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
        c = Categorical(self.model(state))
        action = c.sample()
        return action.item(), c.log_prob(action).view(1)


class NNPolicyAV(BufferedLearner, Policy):
    def __init__(self, model: nn.Module, updater: Updater):
        super().__init__(updater)
        self.model = model

    def __call__(self, state):
        ap, v = self.model(state)
        c = Categorical(ap)
        action = c.sample()
        return action.item(), (c.log_prob(action).view(1), v.item())
