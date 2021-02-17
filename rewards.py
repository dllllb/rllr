import numpy as np
import torch


class ExplicitPosReward:
    def __init__(self):
        self.is_pos_reward = True

    def __call__(self, cur_pos, next_pos, goal_pos):
        cur_dist = np.linalg.norm(cur_pos - goal_pos)
        next_dist = np.linalg.norm(next_pos - goal_pos)

        reward = cur_dist - next_dist
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
            cur_dist = np.abs(cur_pos - goal_pos).sum()
            next_dist = np.abs(next_pos - goal_pos).sum()

        reward = cur_dist - next_dist
        reward = -0.1 if reward == 0 else reward
        return reward


class ExpectedStepsAmountReward:
    def __init__(self, model, max_value=None):
        self.model = model
        self.device = self.model.device
        self.is_pos_reward = False
        self.max_value = max_value

    def _to_torch(self, x):
        return torch.from_numpy(x).float().unsqueeze(0).to(self.device)

    def __call__(self, state, next_state, goal_state):
        self.model.eval()
        with torch.no_grad():
            cur_dist = self.model(self._to_torch(state), self._to_torch(goal_state))
            next_dist = self.model(self._to_torch(next_state), self._to_torch(goal_state))

        if self.max_value is not None:
            cur_dist = torch.clamp(cur_dist, max=self.max_value)
            next_dist = torch.clamp(next_dist, max=self.max_value)

        reward = cur_dist - next_dist
        return reward.cpu().item()


def get_reward_function(conf):
    if conf['training.reward'] == 'explicit_pos_reward':
        return ExplicitPosReward()
    elif conf['training.reward'] == 'explicit_steps_amount':
        return ExplicitStepsAmount()
    elif conf['training.reward'] == 'sparse_pos_reward':
        return SparsePosReward()
    elif conf['training.reward'] == 'sparse_state_reward':
        return SparseStateReward()

    else:
        raise AttributeError(f"unknown reward type '{conf['training.reward']}'")
