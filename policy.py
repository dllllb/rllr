from learner import Learner


class Policy(Learner):
    def update(self, reward):
        pass

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
        return self.actions.sample()
