import numpy as np

import gaming
import mcts


class CheckersGame(gaming.Game):
    def __init__(self, size_x, size_y):
        self.size_x = size_x
        self.size_y = size_y

    def get_possible_actions(self, state, player):
        size_x, size_y = state.shape
        max_shift = min(state.shape)
        figure_pos = np.argwhere(state == player)
        moves = []
        for pos_x, pos_y in figure_pos:
            action = [(pos_x, pos_y)]
            for direction in [-1, 1]:
                for shift in range(1, max_shift):
                    move_x, move_y = pos_x+direction*shift, pos_y+shift
                    if move_x >= size_x or move_y >= size_y:
                        break
                    elif state[move_x, move_y] == player:
                        break
                    elif state[move_x, move_y] != 0:
                        action.append((move_x, move_y))
                    else:
                        action.append((move_x, move_y))
                        moves.append(action)
        return moves

    def get_result_state(self, state, action, player):
        new_state = state.copy()

        for pos_x, pos_y in action[1:-1]:
            new_state[pos_x, pos_y] = 0

        new_state[action[0]] = 0
        new_state[action[-1]] = player

        return new_state

    def get_players(self):
        return [1, 2]

    def get_initial_state(self):
        initial_state = np.zeros((self.size_x, self.size_y), dtype=np.byte)

        for i in range(1):
            for j in range(0, self.size_x, 2):
                initial_state[i, j + i] = 1

        for i in range(self.size_y-1, self.size_y+1):
            for j in range(0, self.size_x, 2):
                initial_state[i, j + i] = 2

        return initial_state

    def get_winner(self, state) -> int:
        if sum(state == 1) == 0:
            return 2
        elif sum(state == 2) == 0:
            return 1
        else:
            return -1


def test_play():
    ttt = CheckersGame(4, 4)
    s1 = mcts.MCTS(ttt, 5)
    s2 = gaming.RandomStrategy(ttt)

    state, winner = gaming.play_game(ttt, [s1, s2])
    print('winner is player N{winner}')
    print(state)


def test_possible_actions():
    ttt = CheckersGame(4, 4)
    board = ttt.get_initial_state()
    actions = ttt.get_possible_actions(board, 1)
    print(actions)
