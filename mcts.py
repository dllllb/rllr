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


class MCTS(gaming.AgentStrategy):
    def __init__(self, game, n_plays: int, max_depth=500):
        self.n_plays = n_plays
        self.max_depth = max_depth
        self.game = game

    def suggest_action(self, root_state, player):
        root = Node(None)

        for i in range(self.n_plays):
            current_node = root
            current_state = root_state
            for j in range(self.max_depth):
                current_node, current_state = self.perform_action(current_node, current_state, player)

                winner = self.game.get_winner(current_state)
                if winner != -1:
                    current_node.update_stats(winner == player)
                    break

                for adversary in self.game.get_players():
                    if adversary != player:
                        current_node, current_state = self.perform_action(current_node, current_state, adversary)

                        winner = self.game.get_winner(current_state)
                        if winner != -1:
                            current_node.update_stats(winner == player)
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


class OneTwoGame(gaming.Game):
    def get_possible_actions(self, state, player):
        return list(range(1, 5))

    def get_result_state(self, state, action, player):
        new_state = state.copy()
        player_state = state[player]
        if player_state == 0 and action in [1, 2]:
            new_state[player] = 1
        elif player_state == 1 and action in [3, 4]:
            new_state[player] = 2

        return new_state

    def get_players(self):
        return [1, 2]

    def get_initial_state(self):
        return {1: 0, 2: 0}

    def get_winner(self, state) -> int:
        for player, state in state.items():
            if state == 2:
                return player
        return -1


def test_mcts_play():
    ttt = OneTwoGame()
    s1 = MCTS(ttt, 50)
    s2 = gaming.RandomStrategy(ttt)

    state, winner, log = gaming.play_game(ttt, [s1, s2], max_turns=50)
    print(f'the winner is the player {winner}')
    print(state)
    print(log)
