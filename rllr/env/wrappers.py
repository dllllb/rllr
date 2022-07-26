import gym
from gym import spaces
import logging
import numpy as np
import torch
import random

from collections import deque, defaultdict
from gym.wrappers import Monitor
import torch.nn.functional as F
from rllr.models import encoders

from ..buffer import ReplayBuffer
from ..exploration import EpisodicMemory
from ..utils import convert_to_torch

from time import sleep
import matplotlib.pyplot as plt
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX, IDX_TO_OBJECT, IDX_TO_COLOR
ACTIONS_TO_IDX = dict(zip(['step_on', 'grab', 'drop'], [i for i in range(3)]))
IDX_TO_ACTIONS = {v: k for k, v in ACTIONS_TO_IDX.items()}

logger = logging.getLogger(__name__)


class TripletHierarchicalWrapper(gym.Wrapper):
    def __init__(self, env, low_level_policy, n_steps=1, objects=None, actions_over_objects=None):
        super(TripletHierarchicalWrapper, self).__init__(env)
        self.objects = objects if objects is not None else ['ball', 'key', 'door', 'box', 'agent']
        self.obj_ids = [OBJECT_TO_IDX[obj] for obj in self.objects]

        if isinstance(env.observation_space, gym.spaces.dict.Dict):
            observation_space = self.observation_space.spaces['image']
        else:
            observation_space = env.observation_space
        self.observation_space = spaces.Dict({'image': observation_space,
                                              'action_mask': spaces.Box(0, 1,
                                                                        shape=(len(self.objects), len(COLOR_TO_IDX)),
                                                                        dtype=np.uint8)})
        self.actions_over_objects = actions_over_objects if actions_over_objects is not None else \
            {
                'box': ['grab'],
                'ball': ['grab'],
                'key': ['grab'],
                'door': ['step_on'],
                'agent': ['drop']
            }
        actions_over_objects = dict()
        for key, value in self.actions_over_objects.items():
            if type(key) == str:
                actions_over_objects[OBJECT_TO_IDX[key]] = value
        self.actions_over_objects.update(actions_over_objects)

        self.policy = low_level_policy
        self.state = None
        self.n_steps = n_steps
        self.action_space = spaces.Dict({
            'object': spaces.Discrete(len(self.objects)),
            'color': spaces.Discrete(len(COLOR_TO_IDX)),
        })

    def get_available_objects(self):
        mask = torch.zeros((len(self.objects), len(COLOR_TO_IDX)), dtype=torch.uint8)
        grid = self.unwrapped.grid.encode()
        xx, yy = np.where(np.isin(grid[:, :, 0], self.obj_ids))
        for obj, c in zip(grid[xx, yy, 0], grid[xx, yy, 1]):
            obj_name = IDX_TO_OBJECT[obj]
            mask[self.objects.index(obj_name), c] = 1
        if self.unwrapped.carrying is not None:
            mask[self.objects.index('agent'), 1] = 1
        return mask

    def observation(self, observation):
        if isinstance(self.env.observation_space, gym.spaces.dict.Dict):
            image = observation['image']
        else:
            image = observation
        return {'image': image,
                'action_mask': self.get_available_objects()}

    def reset(self):
        self.state = self.env.reset()
        self.state = self.observation(self.state)
        return self.state

    def step(self, action):
        cum_reward, step, done, info = 0, 0, False, {}
        obj_type = self.objects[action['object'].squeeze()]
        obj_index = OBJECT_TO_IDX[obj_type]
        color_index = action['color'][0]
        action = self.actions_over_objects[obj_type][0]
        action_index = ACTIONS_TO_IDX[action]

        #print(action, IDX_TO_COLOR[color_index], obj_type)

        action = {
            'object': torch.Tensor([obj_index]).unsqueeze(0),
            'color': torch.Tensor([color_index]).unsqueeze(0),
            'action': torch.Tensor([action_index]).unsqueeze(0)
        }

        goal_emb = self.policy.actor_critic.state_encoder.goal_state_encoder(action)[0]
        while not done and step < self.n_steps:
            _, low_action, _, _ = self.policy.act({
                'image': torch.from_numpy(self.state['image']).unsqueeze(dim=0),
                'goal_emb': goal_emb
            }, None, None, deterministic=True)
            self.state, reward, done, info = self.env.step(low_action)
            self.state = self.observation(self.state)
            cum_reward += reward
            step += 1
            #self.env.render()
        return self.state, cum_reward, done, info


class TripletNavigationWrapper(gym.Wrapper):
    def __init__(self, env, actions=None, objects=None, actions_over_objects=None, choose_objects_uniformly=False):
        super().__init__(env)
        self.choose_objects_uniformly = choose_objects_uniformly
        self.observation_space = spaces.Dict(
            {
                'image': self.observation_space['image'],
                'object': spaces.Discrete(10),
                'color': spaces.Discrete(10),
                'action': spaces.Discrete(10)
            }
        )
        self.is_goal_achieved = False
        self.goal = None
        self.max_goal_steps = 30
        self.goal_steps = 0
        self.prev_obs = None

        self.actions = actions if actions is not None else ['step_on', 'grab', 'drop']
        self.objects = objects if objects is not None else ['ball', 'key', 'door', 'agent', 'box']
        self.objects = [OBJECT_TO_IDX[ob] if type(ob) == str else ob for ob in self.objects]
        self.actions_over_objects = actions_over_objects if actions_over_objects is not None else \
            {
                'ball': ['grab'],
                'key': ['grab'],
                'door': ['step_on'],
                'agent': ['drop'],
                'box': ['grab']
            }
        actions_over_objects = dict()
        for key, value in self.actions_over_objects.items():
            if type(key) == str:
                actions_over_objects[OBJECT_TO_IDX[key]] = value
        self.actions_over_objects.update(actions_over_objects)

    def goal_to_text(self):
        action = self.goal['action']
        object = IDX_TO_OBJECT[self.goal['object']]
        color = IDX_TO_COLOR[self.goal['color']]
        self.unwrapped.mission = ' '.join([action.replace('_', ' '), color, object])

    def set_goal(self, obj, color, action):
        self.goal['object'] = obj
        self.goal['color'] = color
        self.goal['action'] = IDX_TO_ACTIONS[action]
        self.goal_to_text()

    def choose_target_object(self):
        grid = self.unwrapped.grid.encode()
        xx, yy = np.where(np.isin(grid[:, :, 0], self.objects))
        objs = grid[xx, yy, 0]
        unique_objs = set(objs)
        if self.unwrapped.carrying is not None:
            unique_objs.add(OBJECT_TO_IDX['agent'])
        obj_type = random.sample(unique_objs, 1)[0]

        if obj_type == OBJECT_TO_IDX['agent']:
            obj_x, obj_y = self.unwrapped.agent_pos
            return {
                'object': OBJECT_TO_IDX['agent'],
                'color': 1,
                'x': obj_x,
                'y': obj_y
            }

        else:
            if self.choose_objects_uniformly:
                xx, yy = np.where(np.isin(grid[:, :, 0], self.objects))
            else:
                xx, yy = np.where(grid[:, :, 0] == obj_type)
            colors = grid[xx, yy, 1]
            obj_n = np.random.choice(colors.size)
            return {
                'object': grid[xx, yy, 0][obj_n],
                'color': colors[obj_n],
                'x': xx[obj_n],
                'y': yy[obj_n]
            }

    def choose_action(self, obj_type):
        # obj_type = obj_type if type(obj_type)==str else IDX_TO_OBJECT[obj['obj_type']]
        available_actions = self.actions_over_objects[obj_type]
        action = np.random.choice(available_actions)
        return action

    def gen_goal(self):
        while True:
            self.goal = self.choose_target_object()
            self.goal['action'] = self.choose_action(self.goal['object'])
            if not self._goal_achieved():
                self.goal_steps = 0
                self.is_goal_achieved = False
                self.goal_to_text()
                break

    def _goal_achieved(self):
        if self.goal['action'] == 'grab':
            return self.grab_goal_achieved()
        elif self.goal['action'] == 'drop':
            return self.drop_goal_achieved()
        elif self.goal['action'] == 'step_on':
            return self.step_on_goal_achieved()
        else:
            raise NotImplementedError

    def grab_goal_achieved(self):
        cargo = self.unwrapped.carrying
        if cargo is None:
            return False
        else:
            obj_type, obj_color = self.goal['object'], self.goal['color']
            if OBJECT_TO_IDX[cargo.type] == obj_type and COLOR_TO_IDX[cargo.color] == obj_color:
                return True
            else:
                return False

    def drop_goal_achieved(self):
        if self.unwrapped.carrying is None:
            return True
        else:
            return False

    def step_on_goal_achieved(self):
        obj_type, obj_color = self.goal['object'], self.goal['color']
        agent_x, agent_y = self.unwrapped.agent_pos
        grid = self.unwrapped.grid.encode()
        curr_pos_obj, curr_pos_color = grid[agent_x, agent_y, :2]
        if curr_pos_obj == obj_type and curr_pos_color == obj_color:
            return True
        else:
            return False

    def observation(self, obs):
        obs['goal'] = self.goal
        obs['object'] = self.goal['object']
        obs['color'] = self.goal['color']
        obs['action'] = ACTIONS_TO_IDX[self.goal['action']]
        return obs

    def reset(self):
        if self.unwrapped.steps_remaining < self.max_goal_steps or self.prev_obs is None:
            obs = super().reset()
        else:
            obs = self.prev_obs
        self.gen_goal()
        return self.observation(obs)

    def step(self, action):
        obs, env_reward, done, info = self.env.step(action)
        self.is_goal_achieved = self._goal_achieved()
        self.goal_steps += 1
        reward = 1 - 0.9 * (self.goal_steps / self.max_goal_steps) if self.is_goal_achieved else 0

        if env_reward > 0 and done:
            done = False

        if self.is_goal_achieved or (self.goal_steps > self.max_goal_steps):
            done = True
            self.prev_obs = obs

        return self.observation(obs), reward, done, info


class NavigationGoalWrapper(gym.Wrapper):
    """
    Wrapper for navigation through environment to the goal
    """
    def __init__(self, env, goal_achieving_criterion):
        self.goal_state = None
        self.is_goal_achieved = False
        self.goal_achieving_criterion = goal_achieving_criterion
        super().__init__(env)

    def _goal_achieved(self, state):
        if self.goal_state is not None:
            return self.goal_achieving_criterion(state, self.goal_state)
        else:
            return False

    def gen_goal(self, state):
        while True:
            # loop enforces goal_state != current state
            self.gen_goal_(state)
            if not self._goal_achieved(state):
                break

    def gen_goal_(self, state):
        raise NotImplementedError

    def reset(self):
        state = super().reset()
        self.gen_goal(state)
        self.is_goal_achieved = False
        self.curr_steps = 0
        return state

    def step(self, action):
        next_state, env_reward, done, info = self.env.step(action)
        self.is_goal_achieved = self._goal_achieved(next_state)
        reward = int(self.is_goal_achieved)

        if env_reward > 0 and done:
            done = False

        if self.is_goal_achieved and not done:
            self.gen_goal(next_state)

        return next_state, reward, done, info


class RandomGoalWrapper(NavigationGoalWrapper):
    """
    Wrapper for setting random goal
    """
    def __init__(self, env, goal_achieving_criterion, random_goal_generator, verbose=False):
        self.verbose = verbose
        self.random_goal_generator = random_goal_generator
        self.max_steps = 10
        super().__init__(env, goal_achieving_criterion)

    def gen_goal_(self, state):
        #self.goal_state = next(self.random_goal_generator)
        ##
        curr_pos = self.unwrapped.agent_pos
        self.unwrapped.agent_pos = np.random.randint(0, 4, 2) + 1
        self.goal_state, _, _, _ = self.env.step(1)
        self.unwrapped.agent_pos = curr_pos
        self.env.step(0)

    def step(self, action):
        next_state, env_reward, _, info = self.env.step(action)
        self.is_goal_achieved = self._goal_achieved(next_state)
        self.curr_steps += 1

        if self.is_goal_achieved:
            reward = 1 - 0.9 * (self.curr_steps / self.max_steps)
        else:
            reward = 0

        if self.is_goal_achieved or self.curr_steps >= self.max_steps:
            done = True
        else:
            done = False

        return next_state, reward, done, info


class FromBufferGoalWrapper(NavigationGoalWrapper):
    """
    Wrapper for setting goal from buffer
    """
    def __init__(self, env, goal_achieving_criterion, conf, verbose=False, seed=0):
        self.buffer_size = conf['buffer_size']
        self.buffer = deque(maxlen=self.buffer_size)

        if 'unachieved_prob' in conf:
            self.unachieved_prob = conf['unachieved_prob']
            self.unachieved_buffer_size = conf['unachieved_buffer_size']
            self.unachieved_buffer = deque(maxlen=self.unachieved_buffer_size)

        self.warmup_steps = conf['warmup_steps']
        super().__init__(env, goal_achieving_criterion)

        self.count = 0
        self.flag = 0
        self.seed = seed
        np.random.seed(seed)

        self.verbose = 200 if verbose else 0
        if self.verbose:
            self.count = np.random.randint(0, self.verbose)
        else:
            self.count = 0
        self.verbose_episode = False

    def gen_goal(self, state):
        super().gen_goal(state)

        if self.verbose_episode:
            print(self.seed, self.goal_state['position'], self.flag)

    def reset(self):
        self.count += 1
        if self.verbose:
            self.verbose_episode = self.count % self.verbose == 0 and self.goal_state is not None

        if not self.is_goal_achieved and self.goal_state is not None:
            self.unachieved_buffer.append(self.goal_state)

        return super().reset()

    def reset_buffer(self):
        self.buffer = deque(maxlen=self.buffer_size)
        self.unachieved_buffer = deque(maxlen=self.unachieved_buffer_size)

    def buffer_random_choice(self):
        if self.unachieved_buffer and np.random.rand() < self.unachieved_prob:
            choice = np.random.choice(np.arange(len(self.unachieved_buffer)))
            goal_state = self.unachieved_buffer[choice]
            self.flag = 1
        else:
            choice = np.random.choice(np.arange(len(self.buffer)))
            goal_state = self.buffer[choice]
            self.flag = 0

        return goal_state

    def gen_goal_(self, state):
        if len(self.buffer) >= self.warmup_steps:
            self.goal_state = self.buffer_random_choice()
        else:
            self.goal_state = None

    def step(self, action):
        next_state, reward, done, info = super().step(action)
        self.buffer.append(next_state)

        return next_state, reward, done, info


class RNDBufferWrapper(NavigationGoalWrapper):

    def __init__(self, env, goal_achieving_criterion, conf, grid_size, device, verbose=1):

        super().__init__(env, goal_achieving_criterion)
        self.verbose = verbose
        self.device = device
        self.warmup_steps = conf['warmup_steps']
        goal_buffer_size = conf['goal_buffer_size']
        replay_buffer_size = conf['replay_buffer_size']
        self.goal_buffer = deque(maxlen=goal_buffer_size)
        learning_rate = conf['learning_rate']
        batch_size = conf['batch_size']
        update_step = conf['update_step']
        target_network = encoders.get_encoder(grid_size, conf['target'])
        predictor_network = encoders.get_encoder(grid_size, conf['predictor'])
        self.target = target_network.to(device)
        self.predictor = predictor_network.to(device)

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size=replay_buffer_size, batch_size=batch_size, device=device)
        self.steps_count = 0
        self.update_step = update_step
        self.explore_steps = conf.get('replay_buffer_size', 10)

    def _learn(self):
        states = list(self.replay_buffer.sample())[0]
        with torch.no_grad():
            targets = self.target(states)
        outputs = self.predictor(states)
        self.optimizer.zero_grad()
        loss = F.mse_loss(outputs, targets)
        loss.backward()
        self.optimizer.step()

    def gen_goal_(self, state, scores=None):
        if len(self.goal_buffer) >= self.warmup_steps:
            if scores is None:
                scores = self._get_scores()
            choice = np.random.choice(np.arange(len(self.goal_buffer)), p=scores)
            self.goal_state = self.goal_buffer[choice]
        else:
            self.goal_state = None

    def gen_goal(self, state):
        scores = self._get_scores()
        while True:
            # loop enforces goal_state != current state
            self.gen_goal_(state, scores)
            if not self._goal_achieved(state):
                break

    def _get_scores(self):
        if len(self.goal_buffer) == 0:
            return None
        states = convert_to_torch([state['image'] for state in self.goal_buffer])
        with torch.no_grad():
            rnd_scores = (self.target(states) - self.predictor(states)).abs().pow(2).sum(1)
        scores = rnd_scores.numpy()
        return scores/sum(scores)

    def step(self, action):
        state, reward, done, info = super().step(action)
        self.goal_buffer.append(state)
        self.replay_buffer.add(state['image'], )
        self.steps_count += 1
        if (self.steps_count % self.update_step) == 0:
            self.steps_count = 0
            if self.replay_buffer.is_enough():
                self._learn()
        return state, reward, done, info

    def reset(self):
        return super().reset()


class GoAndResetGoalGenerator(NavigationGoalWrapper):
    def __init__(self, env, goal_achieving_criterion, go_agent, conf, verbose=False):
        self.verbose = verbose
        super().__init__(env, goal_achieving_criterion)

        self.verbose = 200 if verbose else 0
        if self.verbose:
            self.count = np.random.randint(0, self.verbose)
        else:
            self.count = 0
        self.verbose_episode = False

        self.device = conf.get('device', 'cpu')
        self.go_agent = go_agent
        self.rhs_size = conf.get('rnn_output', 0)
        self.rhs_n_layers = conf.get('rnn_num_layers', 1)
        self.init_rhs = torch.zeros((1, self.rhs_size * self.rhs_n_layers * 2), device=self.device)
        self.masks = torch.ones((1, 1), device=self.device)
        self.go_steps_low = conf.get('go_steps_low', 5)
        self.go_steps_high = conf.get('go_steps_high', 10)
        self.go_deterministic = conf.get('go_deterministic', False)
        self.n_tries = conf.get('go_n_tries', 1)

    def go(self, next_obs):
        rnn_rhs = self.init_rhs
        go_steps = np.random.randint(self.go_steps_low, self.go_steps_high)
        for _ in range(go_steps):

            _, action, _, rnn_rhs = self.go_agent.act(torch.tensor(next_obs['image']).unsqueeze(0),
                                                      rnn_rhs,
                                                      self.masks,
                                                      deterministic=self.go_deterministic)

            self.env.unwrapped.step_count -= 1
            prev_obs = next_obs
            next_obs, reward, done, info = self.env.step(action)
            if done:
                if reward <= 0:
                    next_obs = prev_obs
                    break

        return next_obs

    def reset(self):
        state = self.env.reset()
        self.gen_goal(state)
        self.is_goal_achieved = False
        return state

    def gen_goal(self, state):
        done = True
        for _ in range(self.n_tries):
            self.gen_goal_(state)
            if not self._goal_achieved(state):
                done = False
                break
        return done

    def gen_goal_(self, state):
        curr_pos = self.unwrapped.agent_pos
        curr_dir = self.unwrapped.agent_dir
        self.goal_state = self.go(state)
        self.unwrapped.agent_pos = curr_pos
        self.unwrapped.agent_dir = curr_dir

    def step(self, action):
        next_state, env_reward, done, info = self.env.step(action)
        self.is_goal_achieved = self._goal_achieved(next_state)
        reward = int(self.is_goal_achieved)

        if env_reward > 0 and done:
            done = False

        if self.is_goal_achieved and not done:
            done = done or self.gen_goal(next_state)

        return next_state, reward, done, info


class GoalObsWrapper(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        if isinstance(env.observation_space, gym.spaces.dict.Dict):
            observation_space = self.observation_space.spaces['image']
        else:
            observation_space = env.observation_space
        self.observation_space = gym.spaces.Dict({
            'image': observation_space,
            'goal_state': observation_space,
        })

    def observation(self, obs):
        if isinstance(obs, dict):
            obs = obs['image']

        goal_obs = self.env.goal_state

        if goal_obs is None:
            goal_obs = obs

        if isinstance(goal_obs, dict):
            goal_obs = goal_obs['image']

        return {'image': obs, 'goal_state': goal_obs}


def navigation_wrapper(env, conf, goal_achieving_criterion, random_goal_generator=None, verbose=False):
    goal_type = conf.get('goal_type', None)
    if goal_type == 'random':
        # env with random goal
        env = RandomGoalWrapper(
            env=env,
            goal_achieving_criterion=goal_achieving_criterion,
            random_goal_generator=random_goal_generator,
            verbose=verbose)

    elif goal_type == 'from_buffer':
        # env with goal from buffer
        env = FromBufferGoalWrapper(
            env,
            goal_achieving_criterion,
            conf['from_buffer_choice_params'],
            verbose=verbose,
            seed=conf['seed'])

    elif goal_type == 'rnd_buffer':
        device = conf['device']
        if conf['env_type'] == 'gym_minigrid':
            grid_size = conf['grid_size'] * conf.get('tile_size', 1)
        else:
            raise AttributeError(f"unknown env_type '{conf['env_type']}'")
        env = RNDBufferWrapper(env,
                               goal_achieving_criterion,
                               conf['rnd_buffer_params'],
                               grid_size,
                               device,
                               verbose=verbose)

    elif goal_type == "go_and_reset":
        go_agent = torch.load(conf['go_and_reset_params.go_agent'], map_location='cpu')
        go_agent.to(conf.get('go_and_reset_params.device', 'cpu'))
        env = GoAndResetGoalGenerator(
            env=env,
            goal_achieving_criterion=goal_achieving_criterion,
            go_agent=go_agent,
            conf=conf['go_and_reset_params'],
            verbose=verbose)
    else:
        raise AttributeError(f"unknown goal_type '{goal_type}'")

    env = GoalObsWrapper(env)
    return env


class FullyRenderWrapper(gym.Wrapper):
    def render(self, *args, **kwargs):
        """This removes the default visualization of the partially observable field of view."""
        kwargs['highlight'] = False
        return self.unwrapped.render(*args, **kwargs)


def visualisation_wrapper(env, video_path):
    env = FullyRenderWrapper(env)  # removes the default visualization of the partially observable field of view.
    env = Monitor(env, video_path, force=True)
    return env


class IntrinsicEpisodicReward(gym.Wrapper):
    """
    Wrapper for adding Intrinsic Episode Memory Reward
        [Badia et al. (2020)](https://openreview.net/forum?id=Sye57xStvB)
    """
    def __init__(self, env, state_embedder=None, beta=0.3):
        self.beta = beta
        self.episodic_memory = EpisodicMemory()
        self.state_embedder = state_embedder if state_embedder is not None else lambda x: x
        super().__init__(env)

    def reset(self):
        state = super().reset()
        self.episodic_memory.clear()
        self.episodic_memory.add(self.state_embedder(state))
        return state

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        with torch.no_grad():
            embedded_state = self.state_embedder(state)
        intrinsic_reward = self.episodic_memory.compute_reward(embedded_state.unsqueeze(0)).item()
        self.episodic_memory.add(embedded_state)

        return state, reward + intrinsic_reward, done, info


class EpisodeInfoWrapper(gym.Wrapper):
    def __init__(self, env):
        super(EpisodeInfoWrapper, self).__init__(env)
        self.episode_reward = 0
        self.episode_steps = 0
        self.visits_stats = defaultdict(int)

    def reset(self):
        self.episode_reward = 0
        self.episode_steps = 0
        self.visits_stats.clear()
        return self.env.reset()

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.episode_reward += reward
        self.episode_steps += 1
        self.visits_stats[tuple(self.agent_pos)] += 1
        info['visit_stats'] = self.visits_stats
        if done:
            info['episode'] = {
                'task': self.env.unwrapped.spec.id,
                'r': self.episode_reward,
                'steps': self.episode_steps
            }
        return state, reward, done, info


class HierarchicalWrapper(gym.Wrapper):
    def __init__(self, env, low_level_policy, action_shape, n_steps=1):
        super(HierarchicalWrapper, self).__init__(env)
        self.policy = low_level_policy
        self.state = None
        self.n_steps = n_steps
        self.action_space = gym.spaces.Box(-1, 1, action_shape)

    def reset(self):
        self.state = self.env.reset()
        return torch.from_numpy(self.state)

    def step(self, action):
        cum_reward, step, done, info = 0, 0, False, {}
        while not done and step < self.n_steps:
            _, low_action, _, _ = self.policy.act({
                'image': torch.from_numpy(self.state).unsqueeze(dim=0),
                'goal_emb': torch.from_numpy(action).unsqueeze(dim=0)
            }, None, None, deterministic=True)
            self.state, reward, done, info = self.env.step(low_action)
            cum_reward += reward
            step += 1
        return self.state, cum_reward, done, info


class HierarchicalWrapperVAE(HierarchicalWrapper):
    def __init__(self, vae, env, low_level_policy, action_shape, n_steps=1):
        super(HierarchicalWrapper, self).__init__(env)
        self.policy = low_level_policy
        self.state = None
        self.n_steps = n_steps
        self.action_space = gym.spaces.Box(-1, 1, action_shape)
        self.vae = vae

    def reset(self):
        self.state = self.env.reset()
        return self.encode(torch.from_numpy(self.state))

    def encode(self, img):
        return self.vae.encoder(img)

    def decode(self, code):
        return self.vae.decode(code)

    def step(self, action):
        cum_reward, step, done, info = 0, 0, False, {}
        while not done and step < self.n_steps:
            _, low_action, _, _ = self.policy.act({
                'image': torch.from_numpy(self.state).unsqueeze(dim=0),
                'goal_emb': torch.from_numpy(action).unsqueeze(dim=0)
            }, None, None, deterministic=True)
            self.state, reward, done, info = self.env.step(low_action)
            cum_reward += reward
            step += 1
        return self.encode(self.state), cum_reward, done, info


class ZeroRewardWrapper(gym.Wrapper):
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if reward > 0 and done:
            done = False

        return state, 0, done, info


class HashCounterWrapper(gym.Wrapper):
    def __init__(self, env, penalty, conf, rnd=None):
        super(HashCounterWrapper, self).__init__(env)
        self.hashed_states = set()
        self.penalty = penalty
        self.hash_type = conf.get('hash_type', 'simple')
        self.pool_size = conf.get('hash_pool_size', 12)

        if self.hash_type == "rnd":
            self.rnd = rnd
            self.device = conf.get('rnd.device', 'cpu')

    def reset(self):
        obs = self.env.reset()
        self.hashed_states = set()
        self.hashed_states.add(self.get_hash(obs))
        return obs

    def get_hash(self, obs):
        if self.hash_type == 'simple':
            state_h = self.env.unwrapped.hash()
        elif self.hash_type == 'simple_position':
            x, y = self.env.unwrapped.agent_pos
            state_h = str(x) + ';' + str(y)
        elif self.hash_type == "rnd":
            img = torch.tensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                img = torch.permute(img, (0, 3, 1, 2))
                img = torch.max_pool2d(img, self.pool_size)
                state_h = self.rnd(img)
        return state_h

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        state_h = self.get_hash(obs)
        if state_h in self.hashed_states:
            if not (self.hash_type == "coord" and state_h == 4):
                reward -= self.penalty/5
        else:
            self.hashed_states.add(state_h)
            if not (self.hash_type == "coord" and state_h == 4):
                reward += self.penalty
        return obs, reward, done, info
