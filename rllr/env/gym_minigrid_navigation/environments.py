import gym
import logging
import numpy as np

from gym_minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper
logger = logging.getLogger(__name__)


class PosObsWrapper(gym.Wrapper):
    """
    Add agent position and direction to state dict
    """
    def observation(self, obs):
        obs = self.env.observation(obs)
        obs['position'] = self.agent_pos
        obs['direction'] = self.agent_dir
        return obs

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        next_state['position'] = self.unwrapped.agent_pos
        next_state['direction'] = self.unwrapped.agent_dir
        return next_state, reward, done, info

    def reset(self):
        state = self.env.reset()
        state['position'] = self.unwrapped.agent_pos
        state['direction'] = self.unwrapped.agent_dir
        return state


class ImageObsWrapper(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.observation_space.spaces['image']

    def observation(self, obs):
        return obs['image']


def gen_wrapped_env(conf, verbose=False):
    if conf['env_task'] in ['MiniGrid-Empty', 'MiniGrid-Dynamic-Obstacles']:
        env_name = f"{conf['env_task']}-{conf['grid_size']}x{conf['grid_size']}-v0"
    else:
        raise AttributeError(f"unknown env_task '{conf['env_task']}'")

    env = gym.make(env_name)
    if not conf.get('rgb_image', False):
        env = FullyObsWrapper(env)  # Fully observable gridworld using a compact grid encoding
    else:
        env = RGBImgObsWrapper(env, tile_size=conf['tile_size'])  # Fully observed RGB image

    if conf.get('goal_achieving_criterion', None) == 'position' or verbose:
        env = PosObsWrapper(env)
    else:
        env = ImageObsWrapper(env)

    return env


def random_grid_goal_generator(conf, verbose=False):
    env = gen_wrapped_env(conf)
    grid_size = env.unwrapped.grid.encode().shape[0]
    init_pos = np.array([1, 1])

    while True:
        goal_pos = None
        while goal_pos is None or (init_pos == goal_pos).all():
            goal_pos = np.random.randint(1, grid_size - 2, 2)
        goal_dir = np.random.randint(0, 4)
        if verbose:
            logger.info(f"Random goal: position {goal_pos}, direction: {goal_dir}")

        env.unwrapped.agent_pos = goal_pos
        env.unwrapped.agent_dir = goal_dir
        goal_state = env.observation(env.unwrapped.gen_obs())
        yield goal_state