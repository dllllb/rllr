import torch
from torch import nn as nn
import numpy as np
from tqdm import trange
import torch.optim as optim
from rllr.utils.common import convert_to_torch
from rllr.buffer.rollout import RolloutStorage


class ContrastiveStateSimilarity:

    def __init__(self,
                 ssim_network,
                 device="cpu",
                 threshold=0.5,
                 lr=1e-3,
                 radius=5,
                 n_updates=100,
                 batch_size=256,
                 epochs=10,
                 verbose=False):

        self.optimizer = torch.optim.Adam(ssim_network.parameters(), lr=lr)
        self.device = device
        self.ssim_network = ssim_network
        self.threshold=threshold,
        self.bce_loss = nn.BCELoss()
        self.radius = radius
        self.batch_size = batch_size
        self.n_updates = n_updates
        self.epochs = epochs
        self.verbose = verbose
        self.bce_loss = nn.BCELoss()
        self.optimizer = optim.Adam(self.ssim_network.parameters(), lr=lr)

    def balance(self, labels, states1, states2):
        labels = np.asarray(labels)
        pos_indices = np.where(labels==True)[0]
        n_pos = pos_indices.shape[0]
        neg_indices = np.where(labels==False)[0]
        n_neg = neg_indices.shape[0]

        if n_neg > n_pos:
            neg_indices = np.random.choice(neg_indices, n_pos, replace=False)
        elif n_pos > n_neg:
            pos_indices = np.random.choice(pos_indices, n_neg, replace=False)
        indices = np.concatenate([pos_indices, neg_indices], dtype=np.int64)
        return labels[indices], states1[indices], states2[indices]

    def update(self, rollouts: RolloutStorage):
        loss = 0
        for p in trange(rollouts.num_processes):
            states1, states2, labels, weights = [], [], [], []
            n, states = 0, []
            for k, m in enumerate(rollouts.masks[:, p].numpy()):
                states.append(rollouts.obs[k, p, :].squeeze())
                n += 1
                if m == 0:
                    for i in range(n):
                        for j in range(i, n):
                            if i != j:
                                states1.append(states[i])
                                states2.append(states[j])
                                labels.append(int(abs(i - j) <= self.radius))
                    n, states = 0, []
            if len(labels) > 0:

                labels = np.asarray(labels)
                states1, states2 = torch.stack(states1, dim=0), torch.stack(states2, dim=0)
                labels, states1, states2 = self.balance(labels, states1, states2)

                loss += self.optimize(states1.to(self.device),
                                      states2.to(self.device),
                                      torch.Tensor(labels, device=self.device))
        return loss/rollouts.num_processes/self.epochs

    def optimize(self, states1, states2, labels):
        total_loss = 0
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            predicted_labels = self.ssim_network.forward(states1, states2).squeeze()
            loss = self.bce_loss(predicted_labels, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss

    def similarity(self, state1, state2):
        return self.ssim_network(state1, state2)

    def to(self, device):
        self.ssim_network = self.ssim_network.to(device)
        return self
