import torch
import torch.nn as nn
import torch.optim as optim


class IMPPO:
    def __init__(self,
                 actor_critic,
                 im_model,
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

    def act(self, state, deterministic=False):
        with torch.no_grad():
            return self.actor_critic.act(state, deterministic)

    def get_value(self, state):
        with torch.no_grad():
            return self.actor_critic.get_value(state)

    def compute_intrinsic_reward(self, next_obs):
        with torch.no_grad():
            return self.im_model.compute_intrinsic_reward(next_obs)

    def update(self, rollouts, obs_rms):
        ext_advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        im_advantages = rollouts.int_returns[:-1] - rollouts.int_value_preds[:-1]

        ext_coef = 2.
        int_coef = 1.

        advantages = ext_advantages * ext_coef + im_advantages * int_coef
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        int_value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, next_obs_batch, actions_batch, value_preds_batch, int_value_preds_batch, \
                    return_batch, int_return_batch, masks_batch, \
                    old_action_log_probs_batch, adv_targ = sample

                #from matplotlib import pyplot as plt
                #f, axarr = plt.subplots(1, 2)
                #for obs, nobs, act in zip(obs_batch, next_obs_batch, actions_batch):
                #    axarr[0].imshow(obs)
                #    axarr[1].imshow(nobs)
                #    axarr[1].set_title(f'action {act}')
                #    plt.show()


                next_obs_batch = ((next_obs_batch - obs_rms.mean) / torch.sqrt(obs_rms.var)).clip(-5, 5)
                im_loss = self.im_model.compute_loss(next_obs_batch)

                # Reshape to do in a single forward pass for all steps
                (values, int_values), action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(obs_batch, actions_batch)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                # clipped_value_loss:
                #value_pred_clipped = value_preds_batch + \
                #     (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                value_loss = (values - return_batch).pow(2).mean()
                #value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                #value_loss = torch.max(value_losses, value_losses_clipped).mean()

                # clipped_int_value_loss:
                #int_value_pred_clipped = int_value_preds_batch + \
                #     (int_values - int_value_preds_batch).clamp(-self.clip_param, self.clip_param)
                int_value_loss = (int_values - int_return_batch).pow(2).mean()
                #int_value_losses_clipped = (int_value_pred_clipped - int_return_batch).pow(2)
                #int_value_loss = torch.max(int_value_losses, int_value_losses_clipped).mean()

                critic_loss = value_loss + int_value_loss

                self.optimizer.zero_grad()
                (critic_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef + im_loss).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                #print(
                #    f'cl {critic_loss.detach().numpy()}, '
                #    f'al {action_loss.detach().numpy()}, '
                #    f'de {dist_entropy.detach().numpy()}, '
                #    f'iml {im_loss.detach().numpy()}')


                value_loss_epoch += value_loss.item()
                int_value_loss_epoch += int_value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        int_value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, int_value_loss_epoch, action_loss_epoch, dist_entropy_epoch
