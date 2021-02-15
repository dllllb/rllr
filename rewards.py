import numpy as np
import torch


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

    def __call__(self, state, next_state, goal_state):
        reward = np.linalg.norm(next_state - goal_state) / 255
        reward = 1 if reward < 2 else -0.1
        return reward


class ExplicitStepsAmount:
    def __init__(self):
        self.is_pos_reward = True

    def __call__(self, cur_pos, next_pos, goal_pos):
        with torch.no_grad():
            dist1 = np.abs(cur_pos - goal_pos).sum()
            dist2 = np.abs(next_pos - goal_pos).sum()

        reward = dist1 - dist2
        reward = -0.1 if reward == 0 else reward
        return reward


class ExpectedStepsAmount:
    def __init__(self, model):
        self.model = model
        self.device = self.model.device
        self.is_pos_reward = False

    def _to_torch(self, x):
        return torch.from_numpy(x).float().unsqueeze(0).to(self.device)

    def __call__(self, state, next_state, goal_state):
        self.model.eval()
        with torch.no_grad():
            dist1 = self.model(self._to_torch(state), self._to_torch(goal_state)).cpu().item()
            dist2 = self.model(self._to_torch(next_state), self._to_torch(goal_state)).cpu().item()

        reward = dist1 - dist2
        return reward


def get_reward_function(conf, model=None):
    if conf['training.reward'] == 'explicit_pos_reward':
        return ExplicitPosReward()
    elif conf['training.reward'] == 'explicit_steps_amount':
        return ExplicitStepsAmount()
    elif conf['training.reward'] == 'sparse_pos_reward':
        return SparsePosReward()
    elif conf['training.reward'] == 'sparse_state_reward':
        return SparseStateReward()
    elif conf['training.reward'] == 'expected_steps_amount':
        return ExpectedStepsAmount(model)
    else:
        raise AttributeError(f"unknown reward type '{conf['training.reward']}'")
