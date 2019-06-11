class Updater:
    def __call__(self, context_hist, state_hist, reward_hist):
        pass


class Learner:
    def update(self, context, state, reward):
        pass

    def end_episode(self):
        pass


class BufferedLearner(Learner):
    def __init__(self, updater):
        self.updater = updater
        self.context_buffer = list()
        self.state_buffer = list()
        self.reward_buffer = list()

    def update(self, context, state, reward):
        self.context_buffer.append(context)
        self.state_buffer.append(state)
        self.reward_buffer.append(reward)

    def end_episode(self):
        self.updater(self.context_buffer, self.state_buffer, self.reward_buffer)
        self.context_buffer = list()
        self.state_buffer = list()
        self.reward_buffer = list()
