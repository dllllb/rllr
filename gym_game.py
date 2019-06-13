import gym

import gaming
from mcts import DepthMCTS


class GymGame(gaming.Game):
    def __init__(self, env_name: str):
        self.env_name = env_name

    def get_initial_state(self):
        env = gym.make(self.env_name)
        state = env.reset()
        return env, state, list()

    def get_possible_actions(self, state, player):
        env, _, _ = state
        return list(range(env.action_space.n))

    def get_player_count(self):
        return 1

    def get_result_state(self, state, action, player):
        env, _, actions = state
        state, reward, done, _ = env.step(action)
        actions = actions.copy()
        actions.append(action)
        return (env, state, actions), reward, done

    def clone(self, state):
        _, _, actions = state
        actions = actions.copy()
        env, _, _ = self.get_initial_state()
        for a in actions:
            state, reward, done, _ = env.step(a)
        return env, state, actions


def test_play():
    ttt = GymGame('CartPole-v1')
    s1 = DepthMCTS(ttt, n_plays=50, max_depth=30, player=1)

    state, rewards, steps, log = gaming.play_game(ttt, [s1], max_turns=50)
    print()
    print(f'steps: {steps}')
    print(rewards)
    print([a for p, a in log])
