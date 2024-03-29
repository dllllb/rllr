import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from collections import defaultdict


def _flatten_helper(T, N, _tensor):
    if _tensor.__class__.__name__ == 'dict':
        return {key: _tensor[key].view(T * N, *_tensor[key].size()[2:]) for key in _tensor}
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_space, action_space, recurrent_hidden_state_size):
        if obs_space.__class__.__name__ == 'Dict':
            self.obs = {key: torch.zeros(num_steps + 1, num_processes, *obs_space[key].shape) for key in obs_space}
        else:
            self.obs = torch.zeros(num_steps + 1, num_processes, *obs_space.shape)

        if action_space.__class__.__name__ == 'Dict':
            self.actions = dict()
            self.action_log_probs = dict()
            for key in action_space:
                space = action_space[key]
                self.action_log_probs[key] = torch.zeros(num_steps, num_processes, 1)
                if space.__class__.__name__ == 'Discrete':
                    self.actions[key] = torch.zeros(num_steps, num_processes, 1).long()
                elif space.__class__.__name__ == 'Box':
                    self.actions[key] = torch.zeros(num_steps, num_processes, space.shape[0])
                else:
                    raise NotImplementedError(f'{space.__class__.__name__} action space is not implemented')
            torch.zeros(num_steps, num_processes, 1).long()
        elif action_space.__class__.__name__ == 'Discrete':
            self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
            self.actions = torch.zeros(num_steps, num_processes, 1).long()
        elif action_space.__class__.__name__ == 'Box':
            self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
            self.actions = torch.zeros(num_steps, num_processes, action_space.shape[0])
        else:
            raise NotImplementedError(f'{action_space.__class__.__name__} action space is not implemented')

        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.num_processes = num_processes
        self.num_steps = num_steps
        self.step = 0

    def copy_obs(self, value, idx):
        if self.obs.__class__.__name__ == 'dict':
            for key in self.obs:
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

        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.masks = self.masks.to(device)

        if type(self.actions) == dict:
            for key in self.actions:
                self.actions[key] = self.actions[key].to(device)
                self.action_log_probs[key] = self.action_log_probs[key].to(device)
        else:
            self.actions = self.actions.to(device)
            self.action_log_probs = self.action_log_probs.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs, value_preds, rewards, masks):
        self.copy_obs(obs, self.step + 1)
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        if type(self.actions) == dict:
            for key in self.actions:
                self.actions[key][self.step].copy_(actions[key])
                self.action_log_probs[key][self.step].copy_(action_log_probs[key])
        else:
            self.actions[self.step].copy_(actions)
            self.action_log_probs[self.step].copy_(action_log_probs)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.copy_obs(self.get_last_obs(), 0)
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
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

            if type(self.actions) == dict:
                actions_batch = {key: self.actions[key].view(-1, self.actions[key].size(-1))[indices]
                                 for key in self.actions}
                old_action_log_probs_batch = {key: self.action_log_probs[key].view(-1, 1)[indices]
                                              for key in self.actions}
            else:
                actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
                old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]

            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]  # UNUSED

            yield obs_batch, actions_batch, recurrent_hidden_states_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            if self.obs.__class__.__name__ == 'dict':
                obs_batch = defaultdict(list)
            else:
                obs_batch = []
            recurrent_hidden_states_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            adv_targ = []

            if type(self.actions) == dict:
                actions_batch = {key: list() for key in self.actions}
                old_action_log_probs_batch = {key: list() for key in self.actions}
            else:
                actions_batch = list()
                old_action_log_probs_batch = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                if self.obs.__class__.__name__ == 'dict':
                    for key in self.obs:
                        obs_batch[key].append(self.obs[key][:-1, ind])
                else:
                    obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(self.recurrent_hidden_states[0:1, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                adv_targ.append(advantages[:, ind])

                if type(self.actions) == dict:
                    for key in actions_batch:
                        actions_batch[key].append(self.actions[key][:, ind])
                        old_action_log_probs_batch[key].append(self.action_log_probs[key][:, ind])
                else:
                    actions_batch.append(self.actions[:, ind])
                    old_action_log_probs_batch.append(self.action_log_probs[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            if self.obs.__class__.__name__ == 'dict':
                obs_batch = {key: torch.stack(obs_batch[key], 1) for key in self.obs}
            else:
                obs_batch = torch.stack(obs_batch, 1)

            if type(self.actions) == dict:
                actions_batch = {key: torch.stack(value, 1) for key, value in actions_batch.items()}
                old_action_log_probs_batch = {key: torch.stack(value, 1) for key, value in old_action_log_probs_batch.items()}
            else:
                actions_batch = torch.stack(actions_batch, 1)
                old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)

            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, actions_batch, recurrent_hidden_states_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
