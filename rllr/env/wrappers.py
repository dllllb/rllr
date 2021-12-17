import gym
import logging
import numpy as np
import torch

from collections import deque
from gym.wrappers import Monitor

from ..exploration import EpisodicMemory

logger = logging.getLogger(__name__)


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
        return state

    def step(self, action):
        next_state, _, done, info = self.env.step(action)
        self.is_goal_achieved = self._goal_achieved(next_state)
        reward = int(self.is_goal_achieved)

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
        super().__init__(env, goal_achieving_criterion)

    def gen_goal_(self, state):
        self.goal_state = next(self.random_goal_generator)


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

        self.verbose = verbose
        self.warmup_steps = conf['warmup_steps']
        super().__init__(env, goal_achieving_criterion)

        self.count = 0
        self.flag = 0
        self.verbose = 200
        self.seed = seed
        np.random.seed(seed)
        self.count = np.random.randint(0, self.verbose)
        self.verbose_episode = False

    def gen_goal(self, state):
        super().gen_goal(state)

        if self.verbose_episode:
            print(self.seed, self.goal_state['position'], self.flag)

    def reset(self):
        self.count += 1
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


class GoalObsWrapper(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        if isinstance(env.observation_space, gym.spaces.dict.Dict):
            observation_space = self.observation_space.spaces['image']
        else:
            observation_space = env.observation_space
        self.observation_space = gym.spaces.Dict({
            'state': observation_space,
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

        return {'state': obs, 'goal_state': goal_obs}


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
        self.visits_stats = dict()

    def reset(self):
        self.episode_reward = 0
        self.episode_steps = 0
        self.visits_stats = dict()
        return self.env.reset()

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.episode_reward += reward
        self.episode_steps += 1
        pos = tuple(self.agent_pos)
        if pos in self.visits_stats:
            self.visits_stats[pos] += 1
        else:
            self.visits_stats[pos] = 1
        info['visit_stats'] = self.visits_stats
        if done:
            info['episode'] = {'r': self.episode_reward, 'steps': self.episode_steps}
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
        cum_reward, step, done = 0, 0, False
        while not done and step < self.n_steps:
            _, low_action, _ = self.policy.act({
                'state': torch.from_numpy(self.state).unsqueeze(dim=0),
                'goal_emb': torch.from_numpy(action).unsqueeze(dim=0)
            }, deterministic=True)
            self.state, reward, done, info = self.env.step(low_action)
            cum_reward += reward
            step += 1
        return self.state, cum_reward, done, info
