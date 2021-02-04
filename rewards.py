import numpy as np
import torch

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
        self.device = torch.device(conf['device'])
        self.resnet.to(self.device)

    def _to_torch(self, arr):
        arr = np.vstack([np.expand_dims(x, axis=0) for x in arr])
        return torch.from_numpy(arr).float().to(self.device).permute(0, 3, 1, 2)

    def __call__(self, state, next_state, goal_state):
        with torch.no_grad():
            embs = self.resnet(self._to_torch([state, next_state, goal_state])).cpu().numpy()

        state_emb, next_state_emb, goal_state_emb = embs
        dist1 = np.linalg.norm(state_emb - goal_state_emb)
        dist2 = np.linalg.norm(next_state_emb - goal_state_emb)

        reward = (dist1 - dist2)
        return reward


def get_reward_function(conf):
    if conf['training.reward'] == 'explicit_pos_reward':
        return explicit_pos_reward
    elif conf['training.reward'] == 'image_net_similarity':
        return ImageNetSimilarity(conf['training.reward_params'])
    else:
        raise AttributeError(f"unknown reward type '{conf['training.reward']}'")
