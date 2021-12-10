import gym
import logging
import numpy as np

from gym_minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, RGBImgPartialObsWrapper


logger = logging.getLogger(__name__)


class PosObsWrapper(gym.core.ObservationWrapper):
    """
    Add agent position and direction to state dict
    """
    def observation(self, obs):
        obs['position'] = np.array([*self.agent_pos, self.agent_dir])
        return obs


class ImageObsWrapper(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.observation_space.spaces['image']

    def observation(self, obs):
        return obs['image']


class RandomStartPointWrapper(gym.Wrapper):
    def __init__(self, env, conf, verbose=False):
        env.seed(conf.get('random_goal_seed', 0))
        self.grid_size = env.unwrapped.grid.encode().shape[0]
        self.verbose = verbose
        super().__init__(env)

    def reset(self):
        super().reset()

        goal_pos = np.random.randint(1, self.grid_size - 1, 2)
        goal_dir = np.random.randint(0, 4)
        if self.verbose:
            logger.info(f"Random goal: position {goal_pos}, direction: {goal_dir}")

        self.env.unwrapped.agent_pos = goal_pos
        self.env.unwrapped.agent_dir = goal_dir
        return self.gen_obs()


class ChangeActionSizeWrapper(gym.Wrapper):
    def __init__(self, env, action_size):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(action_size)


def gen_wrapped_env(conf, verbose=False):
    if conf['env_task'] in ['MiniGrid-Empty', 'MiniGrid-Dynamic-Obstacles']:
        env_name = f"{conf['env_task']}-{conf['grid_size']}x{conf['grid_size']}-v0"

    elif conf['env_task'] == 'MiniGrid-MultiRoom':
        env_names_dict = {
            2: 'MiniGrid-MultiRoom-N2-S4-v0',
            4: 'MiniGrid-MultiRoom-N4-S5-v0',
            6: 'MiniGrid-MultiRoom-N6-v0'
        }
        env_name = env_names_dict[conf['num_rooms']]

    elif conf['env_task'] == 'MiniGrid-FourRooms':
        env_name = 'MiniGrid-FourRooms-v0'

    else:
        raise AttributeError(f"unknown env_task '{conf['env_task']}'")

    env = gym.make(env_name)
    if conf.get('random_start_pos', False):
        env = RandomStartPointWrapper(env, conf, verbose)

    if conf.get('action_size', None) and conf['action_size'] != env.action_space.n:
        env = ChangeActionSizeWrapper(env, conf['action_size'])

    if conf.get('deterministic', True):
        seed = conf.get('seed', 42)
        env.action_space.np_random.seed(seed)
        env.seed(seed)

    if conf.get('fully_observed', True):
        if conf.get('rgb_image', False):
            env = RGBImgObsWrapper(env, tile_size=conf['tile_size'])  # Fully observed RGB image
        else:
            env = FullyObsWrapper(env)  # Fully observable gridworld using a compact grid encoding

    else:
        if conf.get('rgb_image', False):
            env = RGBImgPartialObsWrapper(env, tile_size=conf['tile_size'])  # Fully observed RGB image

    if conf.get('goal_achieving_criterion', None) in {'position_and_direction', 'position'} or verbose:
        env = PosObsWrapper(env)
    else:
        env = ImageObsWrapper(env)

    return env


def random_grid_goal_generator(conf, verbose=False):
    env = gen_wrapped_env({**conf, 'random_start_pos': True}, verbose=verbose)
    env.seed(conf['random_goal_seed'])

    while True:
        goal_state = env.reset()
        yield goal_state
