import gym

import gaming
from mcts import MCTS


class GymGame(gaming.Game):
    def __init__(self, env_name: str):
        self.env_name = env_name

    def get_initial_state(self):
        env = gym.make(self.env_name)
        state = env.reset()
        return env, state

    def get_possible_actions(self, state, player):
        env, _ = state
        return list(range(env.action_space.n))

    def get_player_count(self):
        return 1

    def get_result_state(self, state, action, player):
        env, _ = state
        state, reward, done, _ = env.step(action)
        return (env, state), reward, done


def test_play():
    ttt = GymGame('CartPole-v1')
    s1 = MCTS(ttt, 50, player=1)

    state, rewards, log = gaming.play_game(ttt, [s1], max_turns=50)
    print()
    print(rewards)
    print(log)
