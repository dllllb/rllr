import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_space, action_space):
        if obs_space.__class__.__name__ == 'Dict':
            self.obs = {key: torch.zeros(num_steps + 1, num_processes, *obs_space[key].shape) for key in obs_space}
        else:
            self.obs = torch.zeros(num_steps + 1, num_processes, *obs_space.shape)

        if action_space.__class__.__name__ == 'Discrete':
            self.actions = torch.zeros(num_steps, num_processes, 1).long()
        else:
            self.actions = torch.zeros(num_steps, num_processes, action_space.shape[0])

        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.num_processes = num_processes
        self.num_steps = num_steps
        self.step = 0

    def copy_obs(self, value, idx):
        if self.obs.__class__.__name__ == 'dict':
            for key  in self.obs:
                self.obs[key][idx].copy_(value[key])
        else:
            self.obs[idx].copy_(value)

    def set_first_obs(self, value):
        self.copy_obs(value, 0)

    def get_last_obs(self):
        if self.obs.__class__.__name__ == 'dict':
            return {key: self.obs[key][-1] for key in self.obs}
        else:
            return self.obs[-1]

    def to(self, device):
        if self.obs.__class__.__name__ == 'dict':
            for key in self.obs:
                self.obs[key] = self.obs[key].to(device)
        else:
            self.obs = self.obs.to(device)

        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)

    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks):
        self.copy_obs(obs, self.step + 1)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.copy_obs(self.obs[-1], 0)
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, gamma, gae_lambda):
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.size(0))):
            delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + self.value_preds[step]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            if self.obs.__class__.__name__ == 'dict':
                obs_batch = {key: self.obs[key][:-1].view(-1, *self.obs[key].size()[2:])[indices] for key in self.obs}
            else:
                obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]

            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
