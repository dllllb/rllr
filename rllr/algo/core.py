from abc import ABC
from abc import abstractmethod


class Algo(ABC):

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def reset_episode(self):
        pass

    def train(self, env, n_steps, verbose=False):
        state = env.reset()
        scores, steps_per_episode = [], []
        steps, episode, score = 0, 0, 0

        for step in range(n_steps):
            steps += 1
            action = self.act(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            self.update(state, action, reward, next_state, done)
            state = next_state

            if done:
                episode += 1
                steps_per_episode.append(steps)
                scores.append(score)
                steps, score = 0, 0
                self.reset_episode()
                state = env.reset()

            if verbose and len(scores) > 0 and len(scores) % int(verbose) == 0:
                avg_score = sum(scores) / int(verbose)
                avg_step = sum(steps_per_episode) / int(verbose)
                scores, steps_per_episode = [], []
                print(f"Step: {step}, episodes {episode}. Scores: {avg_score:.2f}, "
                      f"avg. steps per episode: {avg_step:.2f}")

        env.close()
        return True
