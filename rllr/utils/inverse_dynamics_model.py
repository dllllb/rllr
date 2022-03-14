import torch
from torch import nn as nn
import numpy as np
from tqdm import trange
import torch.optim as optim
from rllr.utils.common import convert_to_torch
from rllr.buffer.rollout import RolloutStorage


class IDM:

    def __init__(self,
                 idm_network,
                 device="cpu",
                 lr=1e-3,
                 batch_size=256,
                 epochs=10,
                 verbose=False):

        self.device = device
        self.idm_network = idm_network
        self.ce_loss = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.optimizer = optim.Adam(self.idm_network.parameters(), lr=lr)

    def update(self, rollouts: RolloutStorage):
        loss = 0
        states1, states2, labels = [], [], []
        for p in trange(rollouts.num_processes):
            actions = rollouts.actions[:, p].squeeze()
            states = rollouts.obs[:, p].squeeze()
            mask = rollouts.masks[:, p].squeeze()
            for k, m in enumerate(mask[:-1]):
                if mask[k+1] != 0:
                    states1.append(states[k])
                    states2.append(states[k+1])
                    labels.append(actions[k])
        if len(labels) > 0:
            loss += self.optimize(torch.stack(states1, dim=0).to(self.device),
                                  torch.stack(states2, dim=0).to(self.device),
                                  torch.stack(labels, dim=0).to(self.device))
        return loss/self.epochs

    def optimize(self, states1, states2, labels):
        total_loss = 0
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            predicted_labels = self.idm_network.forward(states1, states2).squeeze()
            loss = self.ce_loss(predicted_labels, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss

    def encode(self, state):
        return self.idm_network.encode(state)

    def to(self, device):
        self.idm_network = self.idm_network.to(device)
        return self
