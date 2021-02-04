import numpy as np

import torchvision.models as zoo_models


def explicit_pos_reward(cur_pos, next_pos, goal_pos):
    dist1 = np.linalg.norm(cur_pos - goal_pos)
    dist2 = np.linalg.norm(next_pos - goal_pos)

    reward = (dist1 - dist2)
    reward = 1 - np.sqrt(2) if reward == 0 else reward
    return reward


class ImageNetSimilarity:
    def __init__(self, conf):
        self.resnet = zoo_models.resnet18(pretrained=conf['pretrained'])

    def __call__(self, state, next_state, goal_state):
        pass


def get_reward_function(conf):
    if conf['training.reward'] == 'explicit_pos_reward':
        return explicit_pos_reward
    elif conf['training.reward'] == 'image_net_similarity':
        return ImageNetSimilarity(conf['training.reward_params'])
    else:
        raise AttributeError(f"unknown reward type '{conf['training.reward']}'")
