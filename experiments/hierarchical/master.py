import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import numpy as np

from rllr.models.ppo import FixedNormal
from vae import VAE


def init_params(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias, 0)


class MasterPolicyModel(nn.Module):

    def __init__(self, state_shape, action_size):
        super(MasterPolicyModel, self).__init__()
        self.state_shape = state_shape
        self.is_recurrent = False
        self.logstd = nn.Parameter(torch.zeros((action_size,)))

        self.action_size = action_size
        self.vae = VAE(state_shape, emb_size=self.action_size)

        self.policy = nn.Sequential(
            nn.Linear(in_features=self.action_size, out_features=self.action_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=self.action_size, out_features=self.action_size)
        )

        self.value = nn.Sequential(
            nn.Linear(in_features=self.action_size, out_features=self.action_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=self.action_size, out_features=1)
        )

        self.apply(init_params)

    def forward(self, inputs, rnn_hxs, masks):
        enc = self.vae.encode(inputs)
        value = self.value(enc)

        mu = self.policy(enc)
        std = self.logstd.exp()
        dist = FixedNormal(mu, std)
        return dist, value, rnn_hxs

    def act(self, states, rnn_hxs, masks, deterministic=False):
        dist, value, rnn_hxs = self.forward(states, rnn_hxs, masks)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        return value, action, dist.log_probs(action), rnn_hxs

    def get_value(self, states, rnn_hxs, masks):
        dist, value, rnn_hxs = self.forward(states, rnn_hxs, masks)
        return value

    def evaluate_actions(self, states, actions, rnn_hxs, masks):
        dist, value, rnn_hxs = self.forward(states, rnn_hxs, masks)
        return value, dist.log_probs(actions), dist.entropy().mean(), rnn_hxs

    def kl_loss(self, states):
        enc = self.vae.encode(states)
        mu = self.policy(enc)
        logvar = self.logstd
        kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_diverge


class MasterPPO:
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def to(self, device):
        self.actor_critic = self.actor_critic.to(device)
        return self

    def act(self, state, rnn_hxs=None, masks=None, deterministic=False):
        with torch.no_grad():
            return self.actor_critic.act(state, rnn_hxs, masks, deterministic)

    def get_value(self, state, rnn_hxs=None, masks=None):
        with torch.no_grad():
            return self.actor_critic.get_value(state, rnn_hxs, masks)

    def update_vae(self, states, batch_size=32):
        rec_loss = []
        for _ in range(states.shape[0] // batch_size):
            ids = torch.randint(0, states.shape[0], (batch_size,))
            imgs = states[ids]
            rec, mu, logvar = self.actor_critic.vae(imgs)

            self.optimizer.zero_grad()
            loss = self.actor_critic.vae.loss(rec, imgs, mu, logvar)
            loss.backward()
            self.optimizer.step()

            rec_loss.append(loss.detach().cpu().numpy())
        return np.mean(rec_loss)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        rec_loss_epoch = 0
        kl_loss_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, \
                    old_action_log_probs_batch, adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, actions_batch, None, None)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                # clipped_value_loss:
                value_pred_clipped = value_preds_batch + \
                     (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                rec, mu, logvar = self.actor_critic.vae(obs_batch)

                self.optimizer.zero_grad()
                ppo_loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
                vae_loss = self.actor_critic.vae.loss(rec, obs_batch, mu, logvar) / 1.e3
                kl_loss = self.actor_critic.kl_loss(obs_batch)

                (ppo_loss + vae_loss + kl_loss).backward()

                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                rec_loss_epoch += vae_loss.item()
                kl_loss_epoch += kl_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        rec_loss_epoch /= num_updates
        kl_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, rec_loss_epoch, kl_loss_epoch
