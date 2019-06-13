import random
from typing import List
import numpy as np

from policy import Policy


class Game:
    def get_possible_actions(self, state, player) -> List[int]:
        pass

    def get_result_state(self, state, action, player):
        return np.array(0)

    def get_players(self) -> List[int]:
        pass

    def get_initial_state(self):
        return np.array(0)

    def get_winner(self, state) -> int:
        pass


class RandomStrategy(Policy):
    def __init__(self, game: Game, player: int):
        self.game = game
        self.player = player

    def __call__(self, state):
        actions = self.game.get_possible_actions(state, self.player)
        return actions[random.randint(0, len(actions)-1)]


def play_game(game: Game, strategies: List[Policy], max_turns=1000):
    state = game.get_initial_state()

    action_log = []
    for i in range(max_turns):
        for player, strategy in zip(game.get_players(), strategies):
            action = strategy(state)
            action_log.append((player, action))
            state = game.get_result_state(state, action, player)

            winner = game.get_winner(state)
            if winner != -1:
                return state, winner, action_log

    return state, 0, action_log  # draw by turns limit
