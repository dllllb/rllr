import gym
import logging
import numpy as np

from collections import deque

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


class NavigationGoalWrapper(gym.Wrapper):
    def __init__(self, env):
        self.goal_state = None
        super().__init__(env)

    def _goal_achieved(self, state):
        # TODO: it is unfair. we do not have position in general case
        return (state['position'] == self.goal_state['position']).all()

    def step(self, action):
        next_state, _, _, info = self.env.step(action)
        done = self.goal_state(next_state) or (self.step_count >= self.max_steps)

        return next_state, 0, done, info


class RandomPosAndGoalWrapper(NavigationGoalWrapper):
    def __init__(self, env, verbose=False):
        self.grid_size = env.unwrapped.grid.encode().shape[0]
        self.verbose = verbose
        super().__init__(env)

    def reset(self):
        # Generate random goal state
        goal_pos = np.random.randint(1, self.grid_size - 2, 2)
        self.unwrapped.agent_pos = goal_pos
        self.goal_state = self.env.observation(self.unwrapped.gen_obs())

        # Set random initial state
        self.env.reset()
        init_pos = None
        while init_pos is None or (init_pos == goal_pos).all():
            init_pos = np.random.randint(1, self.grid_size - 2, 2)
        self.unwrapped.agent_pos = init_pos

        if self.verbose:
            logger.info(f"From {init_pos} to {goal_pos}")

        # Return initial state
        return self.env.observation(self.unwrapped.gen_obs())


class FromBufferGoalWrapper(RandomPosAndGoalWrapper):
    def __init__(self, env, conf, verbose=False):
        self.buffer_size = conf['buffer_size']
        self.buffer = deque(maxlen=self.buffer_size)
        super().__init__(env, verbose)

    def reset(self, is_random=True):
        if is_random:
            super().reset()
        else:
            self.env.reset()

            init_state, goal_state = self.buffer.popleft()
            self.goal_state = goal_state

            self.unwrapped.agent_pos = init_state['position']

            if self.verbose:
                logger.info(f"From {init_state['position']} to {goal_state['position']}")

            return self.env.observation(self.unwrapped.gen_obs())


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
        env = RandomPosAndGoalWrapper(env, verbose=verbose)  # env with random goal and init states
    elif goal_type == 'from_buffer':
        env = FromBufferGoalWrapper(env, conf, verbose=verbose)  # env with goal and init states from buffer
    else:
        raise AttributeError(f"unknown goal_type '{conf['goal_type']}'")

    env = SetRewardWrapper(env, reward_function)  # set reward function

    env = FullyRenderWrapper(env)  # removes the default visualization of the partially observable field of view.
    if conf.get('video_path', False):
        env = Monitor(env, conf['video_path'], force=True)

    env.action_size = action_size

    return env
