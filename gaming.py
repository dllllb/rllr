import random
from typing import List
import numpy as np


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


class AgentStrategy:
    def suggest_action(self, state, player):
        pass


class RandomStrategy:
    def __init__(self, game: Game):
        self.game = game

    def suggest_action(self, state, player):
        actions = self.game.get_possible_actions(state, player)
        return actions[random.randint(0, len(actions)-1)]


def play_game(game: Game, strategies: List[AgentStrategy]):
    state = game.get_initial_state()

    for i in range(10000):
        for player, strategy in zip(game.get_players(), strategies):
            action = strategy.suggest_action(state, player)
            state = game.get_result_state(state, action, player)

            winner = game.get_winner(state)
            if winner != -1:
                return state, winner
