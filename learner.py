class Updater:
    def __call__(self, context_hist, state_hist, reward_hist):
        pass

class MultyEpisodesUpdater:
    def __call__(self, episodes):
        pass

class Learner:
    def update(self, context, state, reward):
        pass

    def end_episode(self):
        pass

class Episode:
    def __init__(self):
        self.context_buffer = list()
        self.state_buffer = list()
        self.reward_buffer = list()

    def update(self, context, state, reward):
        self.context_buffer.append(context)
        self.state_buffer.append(state)
        self.reward_buffer.append(reward)

class BufferedLearner(Learner):
    def __init__(self, updater):
        self.updater = updater
        self.episodes = list()
        self.current_episode = Episode()

    def update(self, context, state, reward):
        self.current_episode.update(context, state, reward)

    def end_episode(self):
        self.episodes.append(self.current_episode)
        self.current_episode = Episode()

    def end(self):
        self.updater(self.episodes)
        self.episodes = list()
        self.current_episode = Episode()
