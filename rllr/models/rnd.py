import torch
import torch.nn as nn
import numpy as np
from torch.nn import init


class RNDModel(nn.Module):
    def __init__(self, predictor, target, device, update_proportion=0.25):
        super(RNDModel, self).__init__()

        self.predictor = predictor.to(device)
        self.target = target.to(device)
        self.device = device
        self.loss = nn.MSELoss(reduction='none')
        self.update_proportion = update_proportion

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False

    def parameters(self):
        return self.predictor.parameters()

    def forward(self, next_obs):
        next_obs = next_obs.float()

        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature

    def compute_intrinsic_reward(self, next_obs):
        with torch.no_grad():
            predict_next_feature, target_next_feature = self.forward(next_obs)
            intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2
            return intrinsic_reward

    def compute_loss(self, next_obs):
        # for Curiosity-driven(Random Network Distillation)
        predict_next_state_feature, target_next_state_feature = self.forward(next_obs)

        forward_loss = self.loss(predict_next_state_feature, target_next_state_feature.detach()).mean(-1)
        # Proportion of exp used for predictor update
        mask = torch.rand(len(forward_loss)).to(self.device)
        mask = (mask < self.update_proportion).type(torch.FloatTensor).to(self.device)
        forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))
        return forward_loss
