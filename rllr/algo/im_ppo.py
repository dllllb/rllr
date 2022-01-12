import torch
import torch.nn as nn
import torch.optim as optim
from rllr.utils.im_training import get_state


class IMPPO:
    def __init__(self,
                 actor_critic,
                 im_model,
                 ext_coef,
                 im_coef,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None):

        self.actor_critic = actor_critic
        self.im_model = im_model

        self.ext_coef = ext_coef
        self.im_coef = im_coef

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

    def act(self, state, rnn_hxs, masks, deterministic=False):
        with torch.no_grad():
            return self.actor_critic.act(state, rnn_hxs, masks, deterministic)

    def get_value(self, state, rnn_hxs, masks):
        with torch.no_grad():
            return self.actor_critic.get_value(state, rnn_hxs, masks)

    def compute_intrinsic_reward(self, next_obs):
        with torch.no_grad():
            return self.im_model.compute_intrinsic_reward(next_obs)

    def update(self, rollouts, obs_rms):
        ext_advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        im_advantages = rollouts.im_returns[:-1] - rollouts.im_value_preds[:-1]

        advantages = ext_advantages * self.ext_coef + im_advantages * self.im_coef
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        im_value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, next_obs_batch, recurrent_hidden_states_batch, actions_batch, value_preds_batch, im_value_preds_batch, \
                    return_batch, im_return_batch, masks_batch, \
                    old_action_log_probs_batch, adv_targ = sample

                next_obs_batch = ((get_state(next_obs_batch) - obs_rms.mean) / torch.sqrt(obs_rms.var)).clip(-5, 5)
                im_loss = self.im_model.compute_loss(next_obs_batch)

                # Reshape to do in a single forward pass for all steps
                (values, im_values), action_log_probs, dist_entropy, rnn_rhs = self.actor_critic.evaluate_actions(
                    obs_batch, actions_batch, recurrent_hidden_states_batch, masks_batch
                )

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                # clipped_value_loss:
                value_pred_clipped = value_preds_batch + \
                     (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()

                # clipped_im_value_loss:
                im_value_pred_clipped = im_value_preds_batch + \
                     (im_values - im_value_preds_batch).clamp(-self.clip_param, self.clip_param)
                im_value_losses = (im_values - im_return_batch).pow(2)
                im_value_losses_clipped = (im_value_pred_clipped - im_return_batch).pow(2)
                im_value_loss = torch.max(im_value_losses, im_value_losses_clipped).mean()

                critic_loss = value_loss + im_value_loss

                self.optimizer.zero_grad()
                (critic_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef + im_loss).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.im_model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                im_value_loss_epoch += im_value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        im_value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, im_value_loss_epoch, action_loss_epoch, dist_entropy_epoch
