import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# TODO: clean it
import numpy as np
import gym
from rllr.env import minigrid_envs
from torch.utils.data import DataLoader, TensorDataset
env_conf = {
    "env_task": "MiniGrid-Empty-8x8-v0",
    "grid_size": 8,
    "action_size": 3,
    "rgb_image": True,
    "tile_size": 8,
    "random_start_pos": True
}
env = minigrid_envs.gen_wrapped_env(env_conf)
def gather_data(n=4096):
    data = list()
    env = minigrid_envs.gen_wrapped_env(env_conf)
    for _ in range(n):
        obs = env.reset()
        data.append(torch.Tensor(obs))
    return data
dataset = gather_data()
dataset = TensorDataset(torch.stack(dataset, dim=0))
data = DataLoader(dataset, batch_size=512, shuffle=True)
def datagen(data):
    while True:
        for batch, in data:
            yield batch
reals_data = datagen(data)


class BCEGANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, is_real):
        if is_real:
            target = torch.ones_like(x)
        else:
            target = torch.zeros_like(x)
        return self.loss(x, target)

gan_loss = BCEGANLoss()


class PPOGAN:
    def __init__(self,
                 actor_critic,
                 encoder,
                 decoder,
                 discriminator,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None):

        self.actor_critic = actor_critic

        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.decoder = decoder
        for p in self.decoder.parameters():
            p.requires_grad = False

        self.discriminator = discriminator

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.agent_optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, eps=eps, betas=(0.5, 0.999))

    def to(self, device):
        self.actor_critic = self.actor_critic.to(device)
        return self

    def act(self, state, rnn_hxs=None, masks=None, deterministic=False):
        with torch.no_grad():
            return self.actor_critic.act(state, rnn_hxs, masks, deterministic)

    def get_value(self, state, rnn_hxs=None, masks=None):
        with torch.no_grad():
            return self.actor_critic.get_value(state, rnn_hxs, masks)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        gan_g_loss = 0
        gan_d_loss = 0
        non_gan_loss = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, actions_batch, recurrent_hidden_states_batch, \
                value_preds_batch, return_batch, masks_batch, \
                old_action_log_probs_batch, adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, actions_batch, recurrent_hidden_states_batch, masks_batch)

                with torch.no_grad():
                    reals = next(reals_data)
                    reals = self.encoder(reals)
                    #reals = self.encoder(obs_batch)
                    fakes = self.actor_critic.deterministic_forward(obs_batch, recurrent_hidden_states_batch, masks_batch)

                for _ in range(1):
                    self.discriminator_optimizer.zero_grad()
                    D_fake = self.discriminator(fakes)
                    D_real = self.discriminator(reals)
                    #D_loss = torch.mean(D_fake - D_real)
                    D_loss = (gan_loss(D_fake, False) + gan_loss(D_real, True)) * 0.5
                    D_loss.backward()
                    self.discriminator_optimizer.step()

                    gan_d_loss += D_loss.item()

                    #for p in self.discriminator.parameters():
                    #    p.data.clamp_(-0.01, 0.01)

                fakes = self.actor_critic.deterministic_forward(obs_batch, recurrent_hidden_states_batch, masks_batch)
                #G_loss = -torch.mean(self.discriminator(fakes))
                G_loss = gan_loss(self.discriminator(fakes), True)

                if type(action_log_probs) == dict:
                    action_loss = 0
                    for key in action_log_probs:
                        ratio = torch.exp(action_log_probs[key] - old_action_log_probs_batch[key])
                        surr1 = ratio * adv_targ
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                        action_loss += -torch.min(surr1, surr2).mean()
                else:
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

                self.agent_optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef + 0.002 * G_loss).backward()
                #nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.agent_optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                gan_g_loss += G_loss.item()
                non_gan_loss += (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).item()


        num_updates = self.ppo_epoch * self.num_mini_batch
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        gan_g_loss /= num_updates
        gan_d_loss /= num_updates
        non_gan_loss /= num_updates

        decoded_img = self.decoder(fakes[:1])

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, decoded_img, gan_g_loss, gan_d_loss, non_gan_loss
