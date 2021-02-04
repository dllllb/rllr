import numpy as np


def explicit_pos_reward(cur_pos, next_pos, goal_pos):
    dist1 = np.linalg.norm(cur_pos - goal_pos)
    dist2 = np.linalg.norm(next_pos - goal_pos)

    reward = (dist1 - dist2)
    reward = 1 - np.sqrt(2) if reward == 0 else reward
    return reward


def get_reward_function(conf):
    if conf['training.reward'] == 'explicit_pos_reward':
        return explicit_pos_reward
    else:
        raise AttributeError(f"unknown reward type '{conf['training.reward']}'")
