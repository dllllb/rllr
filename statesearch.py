import gym


class ExplorationStrategy:
    def update(self, reward):
        pass

    def next_action(self, state):
        pass


class RandomExplorationStrategy(ExplorationStrategy):
    def __init__(self, env):
        self.actions = env.action_space

    def next_action(self, _):
        return self.actions.sample()


def explore_trajectories(env: gym.Env, strategy, n_steps, n_episodes):
    log = []
    for i in range(n_episodes):
        state = env.reset()
        for _ in range(n_steps):
            action = strategy.next_action(state)
            prev_state = state
            state, reward, done, _ = env.step(action)
            strategy.update(reward)
            log.append((prev_state, action, reward, done))
    return log


def test_explore():
    env = gym.make('CartPole-v1')
    strategy = RandomExplorationStrategy(env)
    explore_trajectories(env, strategy, 5, 1)
