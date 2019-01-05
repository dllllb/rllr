import math
import numpy as np

import gaming


class Node:
    def __init__(self, action):
        self.parent = None
        self.action = action
        self.children = []
        self.n_sim = 0
        self.n_win = 0
        self.c = math.sqrt(2)
        self.sf = np.finfo(np.float).tiny

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

        self.children.append(child)

    def get_uct(self):
        base = self.n_win / (self.n_sim + self.sf)
        exp = math.sqrt(2)*math.sqrt(math.log(self.parent.n_sim, math.e)/(self.n_sim+self.sf))
        return base + exp

    def select_child(self):
        return max(e.get_uct() for e in self.children)

    def update_stats(self, win_flag):
        self.n_sim += 1
        self.n_win += win_flag

        if self.parent is not None:
            self.parent.update_stats(win_flag)


class MCTS(gaming.AgentStrategy):
    def __init__(self, game, n_plays: int):
        self.n_plays = n_plays
        self.game = game

    def suggest_action(self, root_state, player):
        root = Node(None)

        for i in range(self.n_plays):
            current = root
            current_state = root_state
            while True:
                if len(current.children) == 0:
                    actions = self.game.get_possible_actions(current_state, player)
                    for action in actions:
                        current.add_child(Node(action))

                current = current.select_child()
                current_state = self.game.get_result_state(current_state, current.action, player)

                winner = self.game.get_winner(current_state)
                if winner != -1:
                    current.update_stats(winner == player)
                    break

                for adversary in self.game.get_players():
                    if adversary != player:
                        adv_action = self.suggest_action(current_state, adversary)
                        current_state = self.game.get_result_state(current_state, adv_action, adversary)

                        winner = self.game.get_winner(current_state)
                        if winner != -1:
                            current.update_stats(winner == player)
                            break

                if winner != -1:
                    break

        best_action = root.select_child().action

        return best_action
