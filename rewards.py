import numpy as np
import torch

import torchvision.models as zoo_models


class ExplicitPosReward:
    def __init__(self):
        self.is_pos_reward = True

    def __call__(self, cur_pos, next_pos, goal_pos):
        dist1 = np.linalg.norm(cur_pos - goal_pos)
        dist2 = np.linalg.norm(next_pos - goal_pos)

        reward = (dist1 - dist2)
        reward = 1 - np.sqrt(2) if reward == 0 else reward
        return reward


class SparsePosReward:
    def __init__(self):
        self.is_pos_reward = True

    def __call__(self, state, next_state, goal_state):
        reward = np.linalg.norm(next_state - goal_state)
        reward = 1 if reward == 0 else -0.1
        return reward


class SparseStateReward:
    def __init__(self):
        self.is_pos_reward = False

    def __call__(self, cur_pos, next_pos, goal_pos):
        reward = np.linalg.norm(next_pos - goal_pos) / 255
        reward = 1 if reward < 2 else -0.1
        return reward


class ResNetSimilarity:
    def __init__(self, conf):
        self.resnet = zoo_models.resnet18(pretrained=conf['pretrained'])
        self.device = torch.device(conf['device'])
        self.resnet.to(self.device)
        self.is_pos_reward = False

    def _to_torch(self, arr):
        arr = np.vstack([np.expand_dims(x, axis=0) for x in arr])
        return torch.from_numpy(arr).float().to(self.device).permute(0, 3, 1, 2)

    def __call__(self, state, next_state, goal_state):
        with torch.no_grad():
            embs = self.resnet(self._to_torch([state, next_state, goal_state])).cpu().numpy()

        state_emb, next_state_emb, goal_state_emb = embs
        state_emb /= np.linalg.norm(state_emb)
        next_state_emb /= np.linalg.norm(next_state_emb)
        goal_state_emb /= np.linalg.norm(goal_state_emb)

        # dist1 = np.linalg.norm(state_emb - goal_state_emb)
        dist2 = np.linalg.norm(next_state_emb - goal_state_emb)

        # reward = (dist1 - dist2)
        print(dist2)
        reward = 1 if dist2 < 1.5 else -0.1
        return reward


def get_reward_function(conf):
    if conf['training.reward'] == 'explicit_pos_reward':
        return ExplicitPosReward()
    elif conf['training.reward'] == 'sparse_pos_reward':
        return SparsePosReward()
    elif conf['training.reward'] == 'sparse_state_reward':
        return SparseStateReward()
    elif conf['training.reward'] == 'res_net_similarity':
        return ResNetSimilarity(conf['training.reward_params'])
    else:
        raise AttributeError(f"unknown reward type '{conf['training.reward']}'")
