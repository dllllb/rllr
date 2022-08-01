import torch
from torch import nn as nn
import numpy as np
import torch.optim as optim


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
                episode_end = reset_ids[end_ids]
                pos_ids_low = np.maximum(target_ids - self.radius, episode_start)
                pos_ids_high = np.minimum(target_ids + self.radius, episode_end)

                bad_ids, = np.where(pos_ids_low >= pos_ids_high)
                target_ids = np.delete(target_ids, bad_ids)
                pos_ids_low = np.delete(pos_ids_low, bad_ids)
                pos_ids_high = np.delete(pos_ids_high, bad_ids)

                pos_ids = np.random.randint(pos_ids_low, pos_ids_high)
                neg_ids = np.random.randint(0, n_observations, pos_ids.shape)
                targets = observations[target_ids]
                positives = observations[pos_ids]
                negatives = observations[neg_ids]
                #from matplotlib import pyplot as plt
                #for i, (trg, pos, neg) in enumerate(zip(targets, positives, negatives)):
                #    f, ax = plt.subplots(1, 3)
                #    ax[0].imshow(trg)
                #    ax[1].imshow(pos)
                #    ax[2].imshow(neg)
                #    st = min(pos_ids[i], target_ids[i])
                #    end = max(pos_ids[i], target_ids[i])
                #    ax[0].set_title(str(dones[st:end + 1]))
                #    plt.show()

                self.optimizer.zero_grad()
                pos_labels = self.ssim_network(targets, positives).squeeze()
                loss = self.bce_loss(pos_labels, torch.ones_like(pos_labels))
                neg_labels = self.ssim_network(targets, negatives).squeeze()
                loss += self.bce_loss(neg_labels, torch.zeros_like(neg_labels))
                loss.backward()
                self.optimizer.step()

                total_loss += loss.detach().cpu().numpy()

        return total_loss / self.epochs

    def similarity(self, state1, state2):
        with torch.no_grad():
            return self.ssim_network(state1, state2)

    def to(self, device):
        self.ssim_network = self.ssim_network.to(device)
        return self
