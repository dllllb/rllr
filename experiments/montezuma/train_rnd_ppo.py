import logging

import gym
from rllr.env.gym_minigrid_navigation import environments as minigrid_envs
from rllr.utils import im_train_ppo
from rllr.env.vec_wrappers import make_vec_envs
from rllr.utils import switch_reproducibility_on, get_conf
from rllr.env.wrappers import EpisodeInfoWrapper
from rllr.models import encoders
from rllr.models.ppo import ActorCriticNetwork
from rllr.algo import IMPPO
from rllr.models import RNDModel
from rllr.utils.logger import init_logger
import torch.nn as nn
from torch.nn import functional as F
from rllr.models.encoders import RNNEncoder
from einops import rearrange
import cv2
import numpy as np
from rllr.models.ppo import FixedCategorical


logger = logging.getLogger(__name__)


class MontezumaInfoWrapper(gym.Wrapper):
    def __init__(self, env):
        super(MontezumaInfoWrapper, self).__init__(env)
        self.room_address = 3
        self.visited_rooms = set()
        self.observation_space = gym.spaces.Box(0, 255, (4, 84, 84), dtype=np.uint8)

        self.episode_reward = 0
        self.episode_steps = 0
        self.last_action = 0

        self.states = None

    def get_current_room(self):
        ram = self.unwrapped.ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        action = int(action)

        # action repeat
        if np.random.rand() < 0.25:
            action = self.last_action

        self.last_action = action

        # max and skip
        last_frames = []
        for t in range(4):
            obs, rew, done, info = self.env.step(action)
            self.episode_reward += np.sign(rew)
            self.episode_steps += 1
            self.visited_rooms.add(self.get_current_room())
            last_frames.append(self.observation(obs))
        obs = np.max(last_frames[-2:], axis=0)

        if done:
            info['episode'] = {
                'r': self.episode_reward,
                'steps': self.episode_steps,
                'task': 'Montezuma',
                'visited_rooms':  len(self.visited_rooms)
            }
            self.visited_rooms.clear()

        self.states = self.states[1:] + [obs]
        return np.asarray(self.states), rew, done, info

    def reset(self):
        self.episode_reward = 0
        self.episode_steps = 0
        self.last_action = 0
        self.states = [np.zeros((84, 84), dtype=np.uint8) for _ in range(3)] + [self.observation(self.env.reset())]
        return np.asarray(self.states)

    def observation(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        return img
        #self.states = self.states[1:] + [img]
        #return np.asarray(self.states)
        # return cv2.resize(t, (84, 84))


class RepeatActionEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.successive_frame = np.zeros((2,) + self.env.observation_space.shape, dtype=np.uint8)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        reward, done = 0, False
        for t in range(4):
            state, r, done, info = self.env.step(action)
            if t == 2:
                self.successive_frame[0] = state
            elif t == 3:
                self.successive_frame[1] = state
            reward += r
            if done:
                break

        state = self.successive_frame.max(axis=0)
        return state, reward, done, info



def gen_env_with_seed(conf, seed):
    env = gym.make(conf['env.env_id'])
    env.seed(seed)
    return MontezumaInfoWrapper(env)



def conv_shape(input, kernel_size, stride, padding=0):
    return (input + 2 * padding - kernel_size) // stride + 1


class PolicyModel(nn.Module):

    def __init__(self, state_shape, n_actions):
        super(PolicyModel, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.is_recurrent = False

        c, w, h = state_shape
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                               stride=1)  # Nature paper -> kernel_size = 3, OpenAI repo -> kernel_size = 4

        conv1_out_w = conv_shape(w, 8, 4)
        conv1_out_h = conv_shape(h, 8, 4)
        conv2_out_w = conv_shape(conv1_out_w, 4, 2)
        conv2_out_h = conv_shape(conv1_out_h, 4, 2)
        conv3_out_w = conv_shape(conv2_out_w, 3, 1)
        conv3_out_h = conv_shape(conv2_out_h, 3, 1)

        flatten_size = conv3_out_w * conv3_out_h * 64

        self.fc1 = nn.Linear(in_features=flatten_size, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=448)

        self.extra_value_fc = nn.Linear(in_features=448, out_features=448)
        self.extra_policy_fc = nn.Linear(in_features=448, out_features=448)

        self.policy = nn.Linear(in_features=448, out_features=self.n_actions)
        self.int_value = nn.Linear(in_features=448, out_features=1)
        self.ext_value = nn.Linear(in_features=448, out_features=1)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()

        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        self.fc1.bias.data.zero_()
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        self.fc2.bias.data.zero_()

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

    def forward(self, inputs):
        x = inputs / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x_v = x + F.relu(self.extra_value_fc(x))
        x_pi = x + F.relu(self.extra_policy_fc(x))
        int_value = self.int_value(x_v)
        ext_value = self.ext_value(x_v)
        logits = self.policy(x_pi)
        dist = FixedCategorical(logits=F.log_softmax(logits, dim=1))

        return dist, int_value, ext_value

    def act(self, states, rnn_hxs, masks, deterministic=False):
        dist, int_value, ext_value = self.forward(states)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        return (int_value, ext_value), action, dist.log_probs(action), rnn_hxs

    def get_value(self, states, rnn_hxs, masks):
        dist, int_value, ext_value = self.forward(states)
        return int_value, ext_value

    def evaluate_actions(self, states, actions, rnn_hxs, masks):
        dist, int_value, ext_value = self.forward(states)
        return (int_value, ext_value), dist.log_probs(actions), dist.entropy().mean(), rnn_hxs

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
    state_conf = config['encoder']

    policy = PolicyModel(env.observation_space.shape, env.action_space.n)

    rnd = RNDModel(
        TargetModel(env.observation_space.shape),
        PredictorModel(env.observation_space.shape),
        config['agent.device'])

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
        config['agent.max_grad_norm']
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
