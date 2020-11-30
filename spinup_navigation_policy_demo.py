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
import torchvision.models as zoo_models
import random

def ssim_l1_dist(srs, dst):
    delta = abs(srs[0] - dst[0]) + abs(srs[1] - dst[1])
    return delta

def ssim_l1_sparce_dist(srs, dst):
    delta = abs(srs[0] - dst[0]) + abs(srs[1] - dst[1])
    return 1 if delta > 0 else 0

def prepare_state(state):
    if isinstance(state, np.ndarray):
        state = torch.from_numpy(state).float()
    if len(state.shape) == 3:
        state = state.view(1, *state.shape)
        state = state.permute(0, 3, 1, 2)
    elif len(state.shape) == 2:
        state = state.view(1, 1, *state.shape)
    elif len(state.shape) == 4:
        state = state.transpose(2, 3).transpose(1, 2)
    return state
    return state.to(DEVICE)

# ------------------ ENV ------------------------ #

class EnvStackObservationsWrapper:

    def __init__(self, env, stack_size=5):

        self.env = env
        self.stack_size = stack_size

        self.obs_space = copy(self.env.observation_space)
        self.obs_space.shape = (self.env.observation_space.shape[0], 
                                self.env.observation_space.shape[1], 
                                self.env.observation_space.shape[2]*stack_size) 

        self.stacked_observations = np.zeros(self.obs_space.shape)

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, **kwargs):
        observation = self.env.reset()
        self.stacked_observations = observation.repeat(self.stack_size, axis=-1)
        return self.stacked_observations

    def step(self, action):  
        observation, reward, done, info = self.env.step(action)
        self.stacked_observations[:, :, :3*(self.stack_size-1)] = self.stacked_observations[:, :, 3:3*self.stack_size]
        self.stacked_observations[:, :, -3:] = observation
        return self.stacked_observations, reward, done, info

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

        self.episodes_actions_buffer = []

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

    @property
    def actions_statistics(self):
        acts = torch.Tensor(self.episodes_actions_buffer)
        acts, acts_cnt = acts.unique(return_counts=True)
        cnt, idx = acts_cnt.sort(descending=True)
        cnt = cnt/cnt.sum()*100
        cnt = cnt.long()
        out = ''
        for e in cnt:
            out += str(e.item()) + ' '
        return out[:-1]

    def generate_new_task(self):
        height = self.env.env.grid.height-1
        width = self.env.env.grid.width-1

        while True:
            # start position
            (x0, y0) = (random.randint(2, width-1), random.randint(2, height-1))

            # end position
            (x1, y1) = (random.randint(2, width-1), random.randint(2, height-1))

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
        self.curent_distance = self.dist_func(self.agent_curent_position, self.agent_goal_position)
        self.min_distance = self.curent_distance

        return observation

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
        self.episodes_actions_buffer.append(action.item())
        return observation, reward, done, info

    def refresh(self):
        self.completed_tasks = 0
        self.succesfully_completed_tasks = 0
        self.episodes_actions_buffer = []

# ------------------ MODEL ---------------------- #

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

def nn_conv_block_raw(img_channels):
    return nn.Sequential(
        nn.Conv2d(img_channels, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=2, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 1, kernel_size=1, stride=1)
    )

def nn_conv_block_vgg(img_channels):
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

'''use in minigrid env for navigation learning'''
class BaseNavModelCONV(nn.Module):
    def __init__(self, env):
        super().__init__()

        with torch.no_grad():
            img_height, img_width, img_channels = env.observation_space.shape
            img_channels = 3

            self.conv = nn_conv_block(img_channels)
            self.conv_target = nn_conv_block(img_channels)
            o = self.conv(torch.zeros(1, img_channels, img_height, img_width))
            self.out_size = 2*int(np.prod(o.size()))

    def forward(self, state):
        current_state, desired_state = torch.chunk(state, chunks=2, dim=-1)
        current_state = prepare_state(current_state).contiguous()
        desired_state = prepare_state(desired_state).contiguous()

        conv_target_out = self.conv_target(desired_state).view(desired_state.size(0), -1)
        conv_out = self.conv(current_state).view(current_state.size(0), -1)
        h = torch.cat((conv_target_out, conv_out), dim=-1)
        return h

        #conv_out = self.conv(current_state).view(current_state.size(0), -1)
        #out = conv_out.flatten(start_dim=1)
        #return out

'''use for most atary games'''
class BaseModelCONV(nn.Module):
    def __init__(self, env):
        super().__init__()

        with torch.no_grad():
            img_height, img_width, img_channels = env.observation_space.shape
            self.conv = nn_conv_block_raw(img_channels)
            self.conv_target = nn_conv_block_raw(img_channels)
            o = self.conv(torch.zeros(1, img_channels, img_height, img_width))
            self.out_size = int(np.prod(o.size()))

    def forward(self, state):
        current_state = prepare_state(state).contiguous()
        conv_out = self.conv(current_state).view(current_state.size(0), -1)
        return conv_out

'''gru cell + conv navigation model'''
class BaseNavModelConvGRU(nn.Module):
    def __init__(self, env, hidden_dim=64):
        super().__init__()
        self.conv_model = BaseNavModelCONV(env)
        self.h_dim = hidden_dim
        self.rnn_cell = nn.GRUCell(self.conv_model.out_size, self.h_dim)
        self.hidden = None

    def reset_hidden(self, device):
        self.hidden = torch.randn(1, self.h_dim).to(device)
   
    def forward(self, state):
        inp = self.conv_model(state)
        self.hidden = self.rnn_cell(inp, self.hidden)
        return self.hidden

class CategoricalActor(nn.Module):
    
    def __init__(self, model):
        super().__init__()
        self.logits_net = model

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(probs=logits)
        #return Categorical(logits=logits)

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
    def __init__(self, env, base_model_type='minigrid_nav'):
        super().__init__()
        self.steps = 0

        if base_model_type == 'minigrid_nav':
            self.body_pi = BaseNavModelCONV(env)
            self.body_v = BaseNavModelCONV(env)
        else:
            self.body_pi = BaseModelCONV(env)
            self.body_v = BaseModelCONV(env)

        h_dim = 256

        self.action_head = nn.Sequential(
            nn.Linear(self.body_pi.out_size, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, env.action_space.n),
            nn.Softmax(dim=-1)
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.body_v.out_size, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1)
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

            self.unfreeze()
        return a.detach().cpu().numpy(), v.detach().cpu().numpy(), logp_a.detach().cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]

    def unfreeze(self):
        self.steps += 1
        if self.steps == 2e5:
            for param in self.parameters():
                param.requires_grad = True
# ----------------------------------------------- #

if __name__ == '__main__':
    # create environment
    env_name = 'minigrid'
    #env_name = 'pong'
    if env_name == 'minigrid':
        env = gym.make('MiniGrid-MyEmptyRandomPosMetaAction-8x8-v0')
        env = RGBImgAndStateObsWrapper(env)
        env = EnvWrapper(env, ssim_l1_dist, allowed_actions_without_reward=10)
    elif env_name == 'pong':
        env = gym.make('Pong-v0')
        env = EnvStackObservationsWrapper(env, stack_size=3)
    else:
        raise NotImplementedError(f'unknown environment: {env_name}')
    env_func = lambda : env

    # create nn model
    device = torch.device('cuda:1')
    base_model_type = 'minigrid_nav' if env_name == 'minigrid' else 'atary'
    ac_kwargs = dict(hidden_sizes=[64,64], activation=nn.Tanh)
    def actor_critic(observation_space, action_space, hidden_sizes, activation):
        return ActorCritic(env, base_model_type).to(device)

    # run
    method = 'ppo'
    if method == 'ppo':
        ppo_pytorch(env_fn=env_func, 
                    actor_critic=actor_critic, 
                    ac_kwargs=ac_kwargs, 
                    steps_per_epoch=1000,
                    train_pi_iters=200,
                    train_v_iters=80,
                    epochs=100000,
                    pi_lr=1e-4,
                    vf_lr=1e-4,
                    device=device,
                    target_kl=0.01)
    elif method == 'vpg':
        vpg_pytorch(env_fn=env_func, 
                    actor_critic=actor_critic, 
                    ac_kwargs=ac_kwargs, 
                    train_v_iters=1,
                    steps_per_epoch=1000,
                    epochs=100000,
                    device=device)
    else:
        print(f'unknown train method: {method}')
    

'''

def generate_circulum_task(self, epoch):
        # for circulum learning
        height = self.env.env.grid.height-1
        width = self.env.env.grid.width-1
        env_size = height + width

        circular_epochs = 100
        val = (max(0, circular_epochs-epoch)**1.4)/circular_epochs
        temperature = math.exp(-val)

        while True:
            n_steps = max(int(temperature*env_size), random.randint(2, 5))

            # start position
            (x0, y0) = (random.randint(2, width-1), random.randint(2, height-1))

            # generate x anywhere between [x0-n_steps, x0+n_steps]
            steps = int(random.random()*n_steps)
            x1 = int(random.randint(x0-steps, x0+steps))
            n_steps = n_steps - abs(x0-x1)

            # generate y like x
            steps = int(random.random()*n_steps)
            y1 = int(random.randint(y0-steps, y0+steps))

            # desired position
            x1, y1 = min(max(2, x1), width-1), min(max(2, y1), height-1)

            if x1!= x0 or y1!=y0:
                break

        # desired state
        self.env.env.agent_start_pos = (x1, y1)
        desired_state = self.env.reset()

        # start state
        self.env.env.agent_start_pos = (x0, y0)
        initial_state = self.env.reset()

        return initial_state, desired_state, abs(x1-x0) + abs(y1-y0)

'''