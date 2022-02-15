from rllr.utils import im_train_ppo
from rllr.env.vec_wrappers import make_vec_envs
from rllr.utils import get_conf
from rllr.algo import IMPPO
from rllr.models import RNDModel
from rllr.utils.logger import init_logger
from rllr.models.encoders import RNNEncoder
from rllr.models.ppo import FixedCategorical
from rllr.utils import gumbel_softmax_sample

import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

from einops.layers.torch import Rearrange
from send_vq_recv import SendRecv

import logging
import gym
from gym.wrappers import TimeLimit
import cv2
import numpy as np
import time

logger = logging.getLogger(__name__)


class StickyActionEnv(gym.Wrapper):
    def __init__(self, env, p=0.25):
        super(StickyActionEnv, self).__init__(env)
        self.p = p
        self.last_action = 0

    def step(self, action):
        if np.random.uniform() < self.p:
            action = self.last_action

        self.last_action = action
        return self.env.step(action)

    def reset(self):
        self.last_action = 0
        return self.env.reset()


class RepeatActionEnv(gym.Wrapper):
    def __init__(self, env, render):
        gym.Wrapper.__init__(self, env)
        self.successive_frame = np.zeros((2,) + self.env.observation_space.shape, dtype=np.uint8)
        self.render = render

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        reward, done = 0, False
        for t in range(4):
            state, r, done, info = self.env.step(action)
            if self.render:
                self.env.render()
                time.sleep(0.01)

            if t == 2:
                self.successive_frame[0] = state
            elif t == 3:
                self.successive_frame[1] = state
            reward += r
            if done:
                break

        state = self.successive_frame.max(axis=0)
        return state, reward, done, info


class MontezumaInfoWrapper(gym.Wrapper):
    def __init__(self, env):
        super(MontezumaInfoWrapper, self).__init__(env)
        self.room_address = 3
        self.visited_rooms = set()
        self.observation_space = gym.spaces.Box(0, 255, (1, 84, 84), dtype=np.uint8)

        self.episode_reward = 0
        self.episode_steps = 0

    def get_current_room(self):
        ram = self.unwrapped.ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        action = int(action)

        obs, rew, done, info = self.env.step(action)
        self.episode_reward += rew
        self.episode_steps += 1
        self.visited_rooms.add(self.get_current_room())

        if done:
            info['episode'] = {
                'r': self.episode_reward,
                'steps': self.episode_steps,
                'task': 'Montezuma',
                'visited_rooms':  len(self.visited_rooms)
            }
            self.visited_rooms.clear()

        return self.observation(obs), np.sign(rew), done, info

    def reset(self):
        self.episode_reward = 0
        self.episode_steps = 0
        return self.observation(self.env.reset())

    def observation(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        return np.stack([img])


def gen_env_with_seed(conf, seed, render=False):
    env = gym.make(conf['env.env_id'])
    env = StickyActionEnv(env)
    env = RepeatActionEnv(env, render)
    env = MontezumaInfoWrapper(env)
    env = TimeLimit(env, max_episode_steps=4500)
    env.seed(seed)
    return env


def conv_shape(input, kernel_size, stride, padding=0):
    return (input + 2 * padding - kernel_size) // stride + 1


class RnnSenderGS(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_dim,
            hidden_size,
            max_len,
            temperature,
            trainable_temperature=False,
    ):
        super(RnnSenderGS, self).__init__()

        assert max_len >= 1, "Cannot have a max_len below 1"
        self.max_len = max_len

        self.f = nn.Linear(256, 256)

        self.hidden_to_output = nn.Linear(hidden_size, vocab_size)

        self.embedding = nn.Linear(vocab_size, embed_dim)
        self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        if not trainable_temperature:
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature]), requires_grad=True
            )

        self.cell = nn.LSTMCell(input_size=embed_dim, hidden_size=hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.sos_embedding, 0.0, 0.01)

    def forward(self, x):
        h_t = self.f(x)

        # return h_t
        c_t = torch.zeros_like(h_t)

        e_t = torch.stack([self.sos_embedding] * h_t.size(0))
        sequence = []

        for step in range(self.max_len):
            h_t, c_t = self.cell(e_t, (h_t, c_t))

            step_logits = self.hidden_to_output(h_t)
            x = gumbel_softmax_sample(
                step_logits, self.temperature, training=True, straight_through=True
            )

            prev_hidden = h_t
            e_t = self.embedding(x)
            sequence.append(x)

        sequence = torch.stack(sequence).permute(1, 0, 2)

        eos = torch.zeros_like(sequence[:, 0, :]).unsqueeze(1)
        eos[:, 0, 0] = 1
        sequence = torch.cat([sequence, eos], dim=1)

        return sequence


class RnnReceiverGS(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super(RnnReceiverGS, self).__init__()

        self.cell = nn.LSTMCell(input_size=embed_dim, hidden_size=hidden_size)
        self.embedding = nn.Linear(vocab_size, embed_dim)
        self.g = nn.Linear(hidden_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, message):
        emb = self.embedding(message)

        h_t, c_t = None, None
        eos_mask = torch.zeros_like(message[:, 0, 0]).unsqueeze(dim=-1)
        outputs = torch.zeros((message.shape[0], self.hidden_size)).to(eos_mask.device)

        # to get an access to the hidden states, we have to unroll the cell ourselves
        for step in range(message.size(1)):
            e_t = emb[:, step, ...]
            h_t, c_t = (
                self.cell(e_t, (h_t, c_t))
                if h_t is not None
                else self.cell(e_t)
            )

            eos_mask = torch.clamp(eos_mask + message[:, step, 0].unsqueeze(dim=-1), 0., 1.)  # always eos == 0
            outputs = outputs * eos_mask + self.g(h_t) * (1.0 - eos_mask)

        return outputs

class LangVQVAE(nn.Module):
    def __init__(self, state_shape, sr, ell2_coef=1e-3, beta=0.25):
        super().__init__()
        self.state_shape = state_shape

        # by default sr is in eval mode for rollout
        self.sr = sr.eval()

        self.output_size = 256
        self.ell2_coef = ell2_coef
        self.beta = beta

        c, w, h = state_shape

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU(inplace=True),

            nn.Flatten(),
            nn.Linear(in_features=3136, out_features=256)
        )

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()
            if isinstance(layer, nn.LSTMCell):
                for name, param in layer.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0)
                    elif 'weight' in name:
                        nn.init.orthogonal_(param)

    def forward(self, t):
        return self.sr(self.enc(t))[0]

    def get_codes(self, t):
        return self.sr(self.enc(t))[1]

    def compute_loss(self, t, batch_size):
        def ell2_reg(x, reduction='sum'):
            return F.mse_loss(x, x.new_zeros(()), reduction=reduction)

        ids = np.random.choice(range(t.shape[0]), (2 * batch_size), replace=False)
        ids_t, ids_d = ids[:batch_size], ids[batch_size:]
        imgs_t = t[ids_t] / 255.
        imgs_d = t[ids_d] / 255.
        feat_t = self.enc(imgs_t)
        feat_d = self.enc(imgs_d)

        # Allow EWMA upadtes to centroids on forward pass thru `sr.vq`
        self.sr.train()
        recout_t, codes, emb_loss, com_loss, ent = self.sr(feat_t, reduction='sum')
        self.sr.eval()

        # \log p(d|t) - \log p(t|t)
        loss = torch.einsum('...f, ...f -> ...',
                            feat_d - feat_t,
                            recout_t)
        
        ell2_d = ell2_reg(feat_d, reduction='sum')
        ell2_t = ell2_reg(feat_t, reduction='sum')
        ell2_r = ell2_reg(recout_t, reduction='sum')

        ell2_loss = 0.5 * (ell2_d + ell2_t + ell2_r)
        hinge_loss = F.relu(1 + loss).sum()

        return (
            hinge_loss + self.ell2_coef * ell2_loss + (emb_loss + self.beta * com_loss)
        ), ent, float(hinge_loss)

class PolicyModel(nn.Module):

    def __init__(self, state_shape, n_actions, encoder):
        super(PolicyModel, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.is_recurrent = True

        self.enc = encoder

        c, w, h = state_shape
        self.rnn = RNNEncoder(self.enc, output_size=448)

        self.extra_value_fc = nn.Linear(in_features=448, out_features=448)
        self.extra_policy_fc = nn.Linear(in_features=448, out_features=448)

        self.policy = nn.Linear(in_features=448, out_features=self.n_actions)
        self.int_value = nn.Linear(in_features=448, out_features=1)
        self.ext_value = nn.Linear(in_features=448, out_features=1)

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        nn.init.orthogonal_(self.extra_policy_fc.weight, gain=np.sqrt(0.1))
        self.extra_policy_fc.bias.data.zero_()
        nn.init.orthogonal_(self.extra_value_fc.weight, gain=np.sqrt(0.1))
        self.extra_value_fc.bias.data.zero_()

        nn.init.orthogonal_(self.policy.weight, gain=np.sqrt(0.01))
        self.policy.bias.data.zero_()
        nn.init.orthogonal_(self.int_value.weight, gain=np.sqrt(0.01))
        self.int_value.bias.data.zero_()
        nn.init.orthogonal_(self.ext_value.weight, gain=np.sqrt(0.01))
        self.ext_value.bias.data.zero_()

    def forward(self, inputs, rnn_hxs, masks):
        inputs = inputs / 255.
        x, rnn_hxs = self.rnn(inputs, rnn_hxs, masks)
        x_v = x + F.relu(self.extra_value_fc(x))
        x_pi = x + F.relu(self.extra_policy_fc(x))
        int_value = self.int_value(x_v)
        ext_value = self.ext_value(x_v)
        logits = self.policy(x_pi)
        dist = FixedCategorical(logits=F.log_softmax(logits, dim=1))

        return dist, int_value, ext_value, rnn_hxs

    def act(self, states, rnn_hxs, masks, deterministic=False):
        dist, ext_value, int_value, rnn_hxs = self.forward(states, rnn_hxs, masks)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        return (ext_value, int_value), action, dist.log_probs(action), rnn_hxs

    def get_value(self, states, rnn_hxs, masks):
        dist, ext_value, int_value, rnn_hxs = self.forward(states, rnn_hxs, masks)
        return ext_value, int_value

    def evaluate_actions(self, states, actions, rnn_hxs, masks):
        dist, ext_value, int_value, rnn_hxs = self.forward(states, rnn_hxs, masks)
        return (ext_value, int_value), dist.log_probs(actions), dist.entropy().mean(), rnn_hxs


class TargetModel(nn.Module):

    def __init__(self, state_shape):
        super(TargetModel, self).__init__()
        self.state_shape = state_shape

        c, w, h = state_shape
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        conv1_out_w = conv_shape(w, 8, 4)
        conv1_out_h = conv_shape(h, 8, 4)
        conv2_out_w = conv_shape(conv1_out_w, 4, 2)
        conv2_out_h = conv_shape(conv1_out_h, 4, 2)
        conv3_out_w = conv_shape(conv2_out_w, 3, 1)
        conv3_out_h = conv_shape(conv2_out_h, 3, 1)

        flatten_size = conv3_out_w * conv3_out_h * 64

        self.encoded_features = nn.Linear(in_features=flatten_size, out_features=512)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()
            elif isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()

    def forward(self, inputs):
        x = inputs / 255.
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu((self.conv3(x)))
        x = x.contiguous()
        x = x.view(x.size(0), -1)

        return self.encoded_features(x)


class PredictorModel(nn.Module):

    def __init__(self, state_shape):
        super(PredictorModel, self).__init__()
        self.state_shape = state_shape

        c, w, h = state_shape
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        conv1_out_w = conv_shape(w, 8, 4)
        conv1_out_h = conv_shape(h, 8, 4)
        conv2_out_w = conv_shape(conv1_out_w, 4, 2)
        conv2_out_h = conv_shape(conv1_out_h, 4, 2)
        conv3_out_w = conv_shape(conv2_out_w, 3, 1)
        conv3_out_h = conv_shape(conv2_out_h, 3, 1)

        flatten_size = conv3_out_w * conv3_out_h * 64

        self.fc1 = nn.Linear(in_features=flatten_size, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.encoded_features = nn.Linear(in_features=512, out_features=512)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()
            elif isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()

    def forward(self, inputs):
        x = inputs / 255.
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu((self.conv3(x)))
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.encoded_features(x)


def get_agent(env, config):
    rnd = RNDModel(
        TargetModel(env.observation_space.shape),
        PredictorModel(env.observation_space.shape),
        config['agent.device'])

    vocab_size = 8  # 100
    max_len = 16  # 3
    hidden_size = 256
    emb_size = 128

    n_words = max_len
    n_symbols = vocab_size
    embedding_dim = 32
    alpha = 0.01

    send = nn.Sequential(
        nn.Linear(256, 256),
        nn.Tanh(),
        nn.Linear(256, n_words * embedding_dim),
        Rearrange('B (M F) -> B M F', M=n_words),
    )

    recv = nn.Sequential(
        Rearrange('B M F -> B (M F)'),
        nn.Linear(n_words * embedding_dim, 256),
        nn.Tanh(),
        nn.Linear(256, 256),
    )

    sr = SendRecv(
        send=send,
        recv=recv,
        num_embeddings=n_symbols,
        embedding_dim=embedding_dim,
        alpha=alpha
    )

    vq = LangVQVAE(env.observation_space.shape, sr)

    policy = PolicyModel(env.observation_space.shape, env.action_space.n, vq)

    return IMPPO(
        policy,
        rnd,
        config['agent.ext_coef'],
        config['agent.im_coef'],
        config['agent.clip_param'],
        config['agent.ppo_epoch'],
        config['agent.num_mini_batch'],
        config['agent.value_loss_coef'],
        config['agent.entropy_coef'],
        config['agent.lr'],
        config['agent.eps'],
        config['agent.max_grad_norm'],
        auto_encoder=vq
    )


def main(args=None):
    config = get_conf(args)

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(config, env_id),
        config['training.n_processes'],
        config['agent.device']
    )

    agent = get_agent(env, config)
    agent.to(config['agent.device'])

    logger.info(f"Running agent training: { config['training.n_steps'] * config['training.n_processes']} steps")
    im_train_ppo(
        env=env,
        agent=agent,
        conf=config,
    )


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('rllr.env.wrappers')
    main()
