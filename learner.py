class Updater:
    def __call__(self, pred_hist, true_hist):
        pass


class Learner:
    def update(self, signal):
        pass

    def end_episode(self):
        pass


class BufferedLearner(Learner):
    def __init__(self, updater):
        self.updater = updater
        self.pred_buffer = list()
        self.signal_buffer = list()

    def update(self, *args):
        if len(args) > 1:
            self.signal_buffer.append(args)
        else:
            self.signal_buffer.append(args[0])

    def end_episode(self):
        self.updater(self.pred_buffer, self.signal_buffer)
        self.pred_buffer = list()
        self.signal_buffer = list()
