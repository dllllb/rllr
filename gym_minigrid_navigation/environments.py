import gym
import logging
import random
import numpy as np

from collections import deque, defaultdict
from functools import partial

from gym.wrappers import Monitor
from gym_minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper

logger = logging.getLogger(__name__)


class FullyRenderWrapper(gym.Wrapper):
    def render(self, *args, **kwargs):
        """This removes the default visualization of the partially observable field of view."""
        kwargs['highlight'] = False
        return self.unwrapped.render(*args, **kwargs)


class PosObsWrapper(gym.Wrapper):
    """
    Add agent pos to state dict
    """
    def observation(self, obs):
        obs = self.env.observation(obs)
        obs['position'] = self.agent_pos
        return obs

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state['position'] = self.unwrapped.agent_pos
        return next_state, reward, done, info

    def reset(self):
        return self.env.reset()['image']


class NavigationGoalWrapper(gym.Wrapper):
    """
    Wrapper for navigation through environment to the goal
    """
    def __init__(self, env):
        self.grid_size = env.unwrapped.grid.encode().shape[0]
        self.init_state = None
        self.goal_state = None
        super().__init__(env)

    def set_state(self, state):
        self.unwrapped.agent_pos = state['position']

    def _set_init_state(self, init_pos):
        self.unwrapped.agent_pos = init_pos
        self.init_state = self.env.observation(self.unwrapped.gen_obs())

    def reset(self):
        # set random init state
        self.env.reset()
        init_pos = np.random.randint(1, self.grid_size - 2, 2)
        self._set_init_state(init_pos)

    def _goal_achieved(self, state):
        if self.goal_state is not None:
            # TODO: it is unfair. we do not have position in general case
            return (state['position'] == self.goal_state['position']).all()
        else:
            return False

    def step(self, action):
        next_state, _, _, info = self.env.step(action)
        is_goal_achieved = self._goal_achieved(next_state)
        done = is_goal_achieved or (self.step_count >= self.max_steps)
        reward = 1 if is_goal_achieved else -0.1

        return next_state, reward, done, info


class RandomGoalWrapper(NavigationGoalWrapper):
    """
    Wrapper for setting random goal
    """
    def __init__(self, env, verbose=False):
        self.verbose = verbose
        super().__init__(env)

    def _generate_state_from_pos(self, pos):
        cur_pos = self.unwrapped.agent_pos
        self.unwrapped.agent_pos = pos
        state = self.env.observation(self.unwrapped.gen_obs())
        self.unwrapped.agent_pos = cur_pos
        return state

    def reset(self):
        super().reset()  # Set random initial state
        init_pos = self.init_state['position']

        # Generate random goal state
        goal_pos = None
        while goal_pos is None or (init_pos == goal_pos).all():
            goal_pos = np.random.randint(1, self.grid_size - 2, 2)
        self.goal_state = self._generate_state_from_pos(goal_pos)

        if self.verbose:
            logger.info(f"From {init_pos} to {goal_pos}")

        # Return initial state
        return self.init_state


class FromBufferGoalWrapper(NavigationGoalWrapper):
    """
    Wrapper for setting goal from buffer
    """
    def __init__(self, env, conf, verbose=False):
        self.buffer_size = conf['buffer_size']
        self.buffer = defaultdict(partial(deque, maxlen=self.buffer_size))
        self.verbose = verbose
        self.complexity = conf['init_complexity']
        self.complexity_step = conf['complexity_step']
        self.threshold = conf['threshold']
        self.update_period = conf['update_period']
        self.max_complexity = conf['max_complexity']
        self.episode_count = 0
        self.done_count = 0
        super().__init__(env)

    def reset_buffer(self):
        self.buffer = defaultdict(partial(deque, maxlen=self.buffer_size))

    def reset(self):
        if self.episode_count % self.update_period == 0:
            self.update_complexity()
        self.episode_count += 1

        complexity = np.random.randint(*self.complexity)
        if len(self.buffer[complexity]) < 100:  # with out goal, only by max_steps episode completion
            super().reset()
            return self.init_state

        else:
            super().reset()
            init_state, goal_state = random.choice(self.buffer[complexity])

            if self.verbose:
                logger.info(f"From {init_state['position']} to {goal_state['position']}")

            self._set_init_state(init_state['position'])
            self.goal_state = goal_state
            return self.init_state

    def step(self, action):
        next_state, reward, done, info = super().step(action)
        if not done:
            self.buffer[self.step_count].append((self.init_state, next_state))
        else:
            self.done_count += 1
        return next_state, reward, done, info

    def update_complexity(self):
        avg_achieved_goals = self.done_count / self.update_period
        self.done_count = 0

        if avg_achieved_goals >= self.threshold and self.complexity[1] + self.complexity_step <= self.max_complexity:
            # self.complexity[0] += self.complexity_step
            self.complexity[1] += self.complexity_step
            logger.info(f"new complexity range: {self.complexity}")


class SetRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_function):
        self.reward_function = reward_function
        self.pos_reward = hasattr(reward_function, 'is_pos_reward') and reward_function.is_pos_reward
        super().__init__(env)

    def step(self, action):
        state = self.env.observation(self.unwrapped.gen_obs())

        next_state, _, done, info = self.env.step(action)

        if self.pos_reward:
            reward = self.reward_function(state['position'], next_state['position'], self.goal_state['position'])
        else:
            reward = self.reward_function(state['image'], next_state['image'], self.goal_state['image'])

        return next_state, reward, done, info


def gen_wrapped_env(conf, reward_function, verbose=False):
    if conf['env_task'] == 'MiniGrid-Empty':
        env_name = f"{conf['env_task']}-{conf['grid_size']}x{conf['grid_size']}-v0"
        action_size = 3
    else:
        raise AttributeError(f"unknown env_task '{conf['env_task']}'")

    env = gym.make(env_name)
    if not conf.get('rgb_image', False):
        env = FullyObsWrapper(env)  # Fully observable gridworld using a compact grid encoding
    else:
        env = RGBImgObsWrapper(env, tile_size=conf['tile_size'])  # Fully observed RGB image

    env = PosObsWrapper(env)

    goal_type = conf.get('goal_type', 'random')
    if goal_type == 'random':
        env = RandomGoalWrapper(env, verbose=verbose)  # env with random goal and init states
    elif goal_type == 'from_buffer':
        env = FromBufferGoalWrapper(env, conf, verbose=verbose)  # env with goal and init states from buffer
    else:
        raise AttributeError(f"unknown goal_type '{conf['goal_type']}'")

    if reward_function is not None:
        env = SetRewardWrapper(env, reward_function)  # set reward function

    env = FullyRenderWrapper(env)  # removes the default visualization of the partially observable field of view.
    if conf.get('video_path', False):
        env = Monitor(env, conf['video_path'], force=True)

    env.action_size = action_size

    return env
