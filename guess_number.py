import numpy as np

import gaming
from mcts import MCTS


class GuessNumber(gaming.Game):
    def __init__(self, max_number=10, n_players=1):
        self.max_number = max_number
        self.n_players = n_players
        self.target = int(np.random.random()*max_number)+1

    def get_possible_actions(self, state, player):
        return list(range(1, self.max_number+1))

    def get_result_state(self, state, action, player):
        state = state.copy()

        if action == self.target:
            state[player] = 0
        elif action < self.target:
            state[player] = -1
        else:
            state[player] = 1

        return state

    def get_players(self):
        return list(range(1, self.n_players+1))

    def get_initial_state(self):
        return dict((p, None) for p in self.get_players())

    def get_winner(self, state) -> int:
        for player, state in state.items():
            if state == 0:
                return player
        return -1


def test_mcts_play():
    ttt = GuessNumber(n_players=2)
    s1 = MCTS(ttt, 50, player=1)
    s2 = gaming.RandomStrategy(ttt, player=2)

    state, winner, log = gaming.play_game(ttt, [s1, s2], max_turns=50)
    print(f'the winner is the player {winner}')
    print(state)
    print(log)


def test_mcts_play_1player():
    ttt = GuessNumber(n_players=1)
    s1 = MCTS(ttt, 50, player=1)

    state, winner, log = gaming.play_game(ttt, [s1], max_turns=50)
    print(f'the winner is the player {winner}')
    print(state)
    print(log)
