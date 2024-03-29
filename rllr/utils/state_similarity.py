import torch
from torch import nn as nn
import numpy as np
import torch.optim as optim


@np.vectorize
def get_neg_ids(episode_start, pos_ids_low, pos_ids_high, episode_end):
    qq = [i for i in range(episode_start, pos_ids_low)] + [i for i in range(pos_ids_high+1, episode_end+1)]
    if len(qq) == 0:
        print(episode_start, pos_ids_low, pos_ids_high, episode_end)
    return np.random.choice([i for i in range(episode_start, pos_ids_low)] +
                            [i for i in range(pos_ids_high+1, episode_end+1)])


class ContrastiveStateSimilarity:

    def __init__(self,
                 ssim_network,
                 lr=1e-3,
                 radius=5,
                 batch_size=256,
                 epochs=10):
        self.ssim_network = ssim_network
        self.bce_loss = nn.BCELoss()
        self.radius = radius
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optim.Adam(self.ssim_network.parameters(), lr=lr)

    def update(self, rollouts):
        observations = rollouts.obs.transpose(1, 0).reshape(-1, *rollouts.obs.shape[2:])
        dones = 1 - rollouts.masks.transpose(1, 0).reshape(-1)

        n_observations = observations.size(0)
        n_batches = n_observations // self.batch_size
        total_loss = 0
        (reset_ids,) = np.where(dones.cpu())
        reset_ids = np.asarray([0] + list(reset_ids) + [n_observations])
        for epoch in range(self.epochs):
            for batch in range(n_batches):
                target_ids = np.random.randint(0, n_observations, (self.batch_size,))

                end_ids = np.searchsorted(reset_ids, target_ids)
                episode_start = reset_ids[end_ids - 1]
                episode_end = reset_ids[end_ids] - 1
                pos_ids_low = np.maximum(target_ids - self.radius, episode_start)
                pos_ids_high = np.minimum(target_ids + self.radius, episode_end)

                bad_ids, = np.where(np.logical_or(pos_ids_low >= pos_ids_high,
                                                  np.logical_and(pos_ids_low == episode_start,
                                                                 pos_ids_high == episode_end)))
                target_ids = np.delete(target_ids, bad_ids)
                pos_ids_low = np.delete(pos_ids_low, bad_ids)
                pos_ids_high = np.delete(pos_ids_high, bad_ids)
                episode_start = np.delete(episode_start, bad_ids)
                episode_end = np.delete(episode_end, bad_ids)

                pos_ids = np.random.randint(pos_ids_low, pos_ids_high+1)
                neg_ids = get_neg_ids(episode_start, pos_ids_low, pos_ids_high, episode_end)
                targets = observations[target_ids]
                positives = observations[pos_ids]
                negatives = observations[neg_ids]

                self.optimizer.zero_grad()
                pos_labels = self.ssim_network(targets, positives).squeeze()
                loss = self.bce_loss(pos_labels, torch.ones_like(pos_labels))
                neg_labels = self.ssim_network(targets, negatives).squeeze()
                loss += self.bce_loss(neg_labels, torch.zeros_like(neg_labels))
                loss.backward()
                self.optimizer.step()

                total_loss += loss.detach().cpu().numpy()

        return total_loss / self.epochs / n_batches

    def similarity(self, state1, state2):
        with torch.no_grad():
            return self.ssim_network(state1, state2)

    def to(self, device):
        self.ssim_network = self.ssim_network.to(device)
        return self
