from constants import *
from custom_envs import *
from custom_wrappers import *

from spinup import ppo_pytorch, vpg_pytorch
import gym
from gym import spaces
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
import spinup.algos.pytorch.ppo.core as core

from copy import copy

from navigation_models import ConvNavPolicy2, state_embed_block__
import torchvision.models as zoo_models
import random

# ------------------ ENV ---------------------- #

env = gym.make('MiniGrid-MyEmptyRandomPosMetaAction-8x8-v0')
env = RGBImgAndStateObsWrapper(env)

env2 = gym.make('Pong-v0')

def ssim_l1_dist(srs, dst):
    delta = abs(srs[0] - dst[0]) + abs(srs[1] - dst[1])
    return delta

def ssim_l1_sparce_dist(srs, dst):
    delta = abs(srs[0] - dst[0]) + abs(srs[1] - dst[1])
    return 1 if delta > 0 else 0

class EnvWrapper:

    def __init__(self, env, 
                dist_func=ssim_l1_dist, 
                allowed_actions_without_reward=10, 
                overide_action_space=True,
                overide_observation_space=True):
        self.env = env
        self.dist_func = dist_func

        self.agent_curent_position = None
        self.agent_curent_image = None
        
        self.agent_goal_position = None
        self.agent_goal_image = None
        self.actions_without_reward = 0
        self.allowed_actions_without_reward = allowed_actions_without_reward

        self.curent_distance = -1
        self.min_distance = -1

        self.succesfully_completed_tasks = 0
        self.completed_tasks = 0

        # we override action spaces
        self.actions = spaces.Discrete(n=4)

        self.observations = copy(self.env.observation_space)
        sh = self.observations.shape
        self.observations.shape = (sh[0], sh[1], 2*sh[2])

        #
        self.x1 = None
        self.y1 = None

        self.overide_action_space = overide_action_space
        self.overide_observation_space = overide_observation_space

    @property
    def observation_space(self):
        if self.overide_observation_space:
            return self.observations
        else:
            return self.env.observation_space

    @property
    def action_space(self):
        if self.overide_action_space:
            return self.actions
        else:
            return self.env.action_space

    @property
    def saccesful_tasks_ratio(self):
        #return self.succesfully_completed_tasks
        return 0 if self.completed_tasks == 0 else round(float(self.succesfully_completed_tasks)/self.completed_tasks, 2)

    def generate_new_task(self):
        height = self.env.env.grid.height-1
        width = self.env.env.grid.width-1

        while True:
            # start position
            (x0, y0) = (random.randint(2, width-1), random.randint(2, height-1))

            # end position
            if True or self.x1 is None:
                (x1, y1) = (random.randint(2, width-1), random.randint(2, height-1))
                self.x1, self.y1 = x1, y1
            else:
                x1, y1 = self.x1, self.y1

            if x1!= x0 or y1!=y0:
                break

        # goal state
        self.env.env.agent_start_pos = (x1, y1)
        desired_state = self.env.reset()
        self.agent_goal_position = desired_state['agent_pos']
        self.agent_goal_image = desired_state['image']

        # start state
        self.env.env.agent_start_pos = (x0, y0)
        initial_state = self.env.reset()
        self.agent_curent_position = initial_state['agent_pos']
        self.agent_curent_image = initial_state['image']

    def reset(self, **kwargs):
        self.actions_without_reward = 0

        self.generate_new_task()

        observation = np.concatenate((self.agent_curent_image, self.agent_goal_image), axis=-1)
        #observation = self.agent_curent_image
        self.curent_distance = self.dist_func(self.agent_curent_position, self.agent_goal_position)
        self.min_distance = self.curent_distance

        return observation
        '''
        observation = self.env.reset(**kwargs)
        observation = self.observation(observation)
        return observation
        '''

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        done = False

        self.agent_curent_position = observation['agent_pos']
        self.agent_curent_image = observation['image']

        self.curent_distance = self.dist_func(self.agent_curent_position, self.agent_goal_position)
        if self.curent_distance < self.min_distance:
            self.min_distance = self.curent_distance
            reward = 1
        else:
            self.actions_without_reward += 1
            reward = 0

        if self.actions_without_reward > self.allowed_actions_without_reward:
            self.completed_tasks += 1
            done = True

        if self.min_distance == 0:
            self.completed_tasks += 1
            self.succesfully_completed_tasks += 1
            done = True

        observation = np.concatenate((self.agent_curent_image, self.agent_goal_image), axis=-1)
        #observation = self.agent_curent_image
        return observation, reward, done, info

    def refresh(self):
        self.completed_tasks = 0
        self.succesfully_completed_tasks = 0

env_func = lambda : EnvWrapper(env, ssim_l1_dist, allowed_actions_without_reward=20)
env_func2 = lambda: env2
# ----------------------------------------------- #

ac_kwargs = dict(hidden_sizes=[64,64], activation=nn.Tanh)

# ------------------ MODEL ---------------------- #

def prepare_state(state):
    if isinstance(state, np.ndarray):
        state = torch.from_numpy(state).float()# / 255
    if len(state.shape) == 3:
        state = state.view(1, *state.shape)
        state = state.permute(0, 3, 1, 2)
    elif len(state.shape) == 2:
        state = state.view(1, 1, *state.shape)
    elif len(state.shape) == 4:
        state = state.transpose(2, 3).transpose(1, 2)
    return state
    return state.to(DEVICE)

def nn_mlp_block(input_shape):
    d_in = np.prod(input_shape)
    return nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(d_in, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU()
    )

def nn_conv_block(img_channels):
    model = torch.load(f'/mnt2/molchanov/models/rl_epoch2')
    model = nn.Sequential(*list(model.children())[:-3])
    for param in model.parameters():
        param.requires_grad = False
    return model.to(torch.device('cpu'))

def nn_conv_block__(img_channels):
    return nn.Sequential(
        nn.Conv2d(img_channels, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=2, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 1, kernel_size=1, stride=1)
    )

def nn_conv_block_(img_channels):
    # get pretrained body
    model = zoo_models.vgg11(pretrained=True, progress=True)
    sub_model =  list(list(model.children())[0].children())
    sub_model = nn.Sequential(*sub_model[:11])
    #for param in sub_model.parameters():
    #    param.requires_grad = False
    return sub_model

class BaseModelMLP(nn.Module):
    def __init__(self, env: gym.Env):
        super().__init__()

        img_height, img_width, img_channels = env.observation_space.shape
        img_channels = 3
        self.model = nn_mlp_block(np.array([img_height, img_width, img_channels]))
        self.out_size = 64

    def forward(self, state):
        current_state, desired_state = torch.chunk(state, chunks=2, dim=-1)
        current_state = prepare_state(current_state)
        #desired_state = prepare_state(desired_state)
        return self.model(current_state)

class BaseModelCONV(nn.Module):
    def __init__(self, env: gym.Env):
        super().__init__()

        with torch.no_grad():
            self.mu = None
            self.sigma = None
            img_height, img_width, img_channels = env.observation_space.shape
            img_channels = 3

            self.conv = nn_conv_block(img_channels)
            self.conv_target = nn_conv_block(img_channels)
            o = self.conv(torch.zeros(1, img_channels, img_height, img_width))
            self.out_size = 2*int(np.prod(o.size()))

            #self.fc = nn.Sequential(
            #    nn.Linear(self.out_size, 2),
            #    nn.Tanh()
            #)

    def forward(self, state):
        current_state, desired_state = torch.chunk(state, chunks=2, dim=-1)
        current_state = prepare_state(current_state).contiguous()
        desired_state = prepare_state(desired_state).contiguous()

        '''
        if not self.mu:
            self.mu = current_state.mean().item()
            self.sigma = current_state.std().item()

        current_state = (current_state - self.mu)/(self.sigma)
        desired_state = (desired_state - self.mu)/(self.sigma)
        '''

        #conv_out = self.conv(current_state).view(current_state.size(0), -1)
        #out = conv_out.flatten(start_dim=1)
        #return out

        conv_target_out = self.conv_target(desired_state).view(desired_state.size(0), -1)
        conv_out = self.conv(current_state).view(current_state.size(0), -1)
        h = torch.cat((conv_target_out, conv_out), dim=-1)
        return h

class CategoricalActor(nn.Module):
    
    def __init__(self, model):
        super().__init__()
        self.logits_net = model

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class ActorCritic(nn.Module):
    def __init__(self, env: gym.Env):
        super().__init__()

        self.body_pi = BaseModelCONV(env)
        self.body_v = BaseModelCONV(env)
        h_dim = 64

        self.action_head = nn.Sequential(
            nn.Linear(self.body_pi.out_size, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 4),#3),#env.action_space.n),
            #nn.LogSoftmax(dim=-1)
        )

        '''
        self.value_head = nn.Sequential(
            nn.Linear(2, 512),#3),#env.action_space.n),
            nn.ReLU(),
            nn.Linear(512, 1),#3),#env.action_space.n),
            nn.Tanh(),
        )
        '''
        self.value_head = nn.Sequential(
            nn.Linear(self.body_v.out_size, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),#3),#env.action_space.n),
        )

        self.pi_model = nn.Sequential(
            self.body_pi,
            self.action_head
        )
        self.v = nn.Sequential(
            self.body_v,
            self.value_head
        )

        self.pi = CategoricalActor(self.pi_model)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.detach().cpu().numpy(), v.detach().cpu().numpy(), logp_a.detach().cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]

def actor_critic(observation_space, action_space, hidden_sizes, activation):
    #obs_space = np.prod(observation_space.shape)
    #obs_space = np.zeros(obs_space)
    #return core.MLPActorCritic(obs_space, action_space, hidden_sizes, activation).to(DEVICE)
    return ActorCritic(env).to(DEVICE)

# ----------------------------------------------- #

if __name__ == '__main__':
    
    ppo_pytorch(env_fn=env_func, 
                actor_critic=actor_critic, 
                ac_kwargs=ac_kwargs, 
                steps_per_epoch=1000,#5000, 
                epochs=1000,
                device=DEVICE)
    

    '''
    vpg_pytorch(env_fn=env_func, 
                actor_critic=actor_critic, 
                ac_kwargs=ac_kwargs, 
                train_v_iters=1,
                steps_per_epoch=500,#5000, 
                epochs=150,
                device=DEVICE)
    '''