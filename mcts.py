import math
import numpy as np

import gaming
from policy import Policy


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

    def get_uct(self):
        base = (self.n_win + 1) / (self.n_sim + 1)
        exp = math.sqrt(2)*math.sqrt(math.log(self.parent.n_sim+1, math.e)/(self.n_sim+1))
        return base + exp

    def select_child(self):
        ucts = np.array([e.get_uct() for e in self.children])
        pos = np.argwhere(ucts == max(ucts))
        selected_child = np.random.randint(0, len(pos))
        return self.children[selected_child]

    def update_stats(self, win_flag):
        self.n_sim += 1
        self.n_win += win_flag

        if self.parent is not None:
            self.parent.update_stats(win_flag)


class MCTS(Policy):
    def __init__(self, game: gaming.Game, n_plays: int, player: int, max_depth=500):
        self.n_plays = n_plays
        self.max_depth = max_depth
        self.game = game
        self.player = player

    def __call__(self, root_state):
        root = Node(None)

        for i in range(self.n_plays):
            current_node = root
            current_state = root_state
            for j in range(self.max_depth):
                current_node, current_state = self.perform_action(current_node, current_state, self.player)

                winner = self.game.get_winner(current_state)
                if winner != -1:
                    current_node.update_stats(winner == self.player)
                    break

                for adversary in self.game.get_players():
                    if adversary != self.player:
                        current_node, current_state = self.perform_action(current_node, current_state, adversary)

                        winner = self.game.get_winner(current_state)
                        if winner != -1:
                            current_node.update_stats(winner == self.player)
                            break

                if winner != -1:
                    break

        best_action = root.select_child().action

        return best_action

    def perform_action(self, node, state, player):
        if len(node.children) == 0:
            actions = self.game.get_possible_actions(state, player)
            for action in actions:
                node.add_child(Node(action))

        if len(node.children) != 0:
            node = node.select_child()
            state = self.game.get_result_state(state, node.action, player)
            return node, state
        else:
            return node, state
