import gym
import logging
import numpy as np
import math

from gym_minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, RGBImgPartialObsWrapper


logger = logging.getLogger(__name__)


class PosObsWrapper(gym.core.ObservationWrapper):
    """
    Add agent position and direction to state dict
    """
    def observation(self, obs):
        obs['position'] = np.array([*self.agent_pos, self.agent_dir])
        return obs


class ImageObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        if type(self.observation_space) == gym.spaces.Dict:
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


class FixResetSeedWrapper(gym.Wrapper):
    def __init__(self, env, seed=0):
        self.seed_value = seed
        super().__init__(env)
        self.reset()

    def reset(self):
        self.env.seed(self.seed_value)
        state = super().reset()
        return state


class GoalPatch(gym.Wrapper):
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        if reward > 0 and done:
            done = False
        return next_state, reward, done, info


class PosEncoding(gym.ObservationWrapper):
    def __init__(self, env):
        assert env.tile_size == 4
        super().__init__(env)
        self.tile_size = env.tile_size
        self.d_model = (env.tile_size ** 2) // 2
        self.div_term = np.tile(np.exp(np.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model)), 7)
        self.pe = np.zeros((7*env.tile_size, 7*env.tile_size))

    def observation(self, obs):
        img_obs = obs['image']

        pos = self.env.agent_pos
        direction = self.env.agent_dir

        if direction == 0:
            xx = np.arange(7) + pos[0]
            yy = np.arange(-3, 4) + pos[1]
        elif direction == 1:
            xx = np.arange(-3, 4) + pos[0]
            yy = np.arange(7) + pos[1]
        elif direction == 2:
            xx = np.arange(-6, 1) + pos[0]
            yy = np.arange(-3, 4) + pos[1]
        else:
            xx = np.arange(-3, 4) + pos[0]
            yy = np.arange(-6, 1) + pos[1]

        xx, yy = xx.repeat(self.tile_size), yy.repeat(self.tile_size)
        xx, yy = np.meshgrid(xx, yy)

        for i in range(0, 7*self.tile_size, self.tile_size):
            self.pe[i, :] = np.sin(xx[i] * self.div_term)
            self.pe[i+1, :] = np.cos(xx[i] * self.div_term)
            self.pe[i+2, :] = np.sin(yy[i] * self.div_term)
            self.pe[i+3, :] = np.cos(yy[i] * self.div_term)

        img_obs += np.expand_dims(self.pe, -1)
        obs['image'] = img_obs
        return obs


class ImgNorm(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        self.observation_space.spaces['image'] = gym.spaces.Box(-5, 5,
                                                                env.observation_space.spaces['image'].shape,
                                                                dtype=np.float32)

    def observation(self, obs):
        img_obs = obs['image']
        img_obs = img_obs / 255.
        img_obs = ((img_obs - self.mean) / self.std).clip(-5, 5)
        obs['image'] = img_obs
        return obs


def get_env_name(conf):
    if conf['env_task'] in ['MiniGrid-Empty', 'MiniGrid-Dynamic-Obstacles']:
        return f"{conf['env_task']}-{conf['grid_size']}x{conf['grid_size']}-v0"

    elif conf['env_task'] == 'MiniGrid-MultiRoom':
        env_names_dict = {
            2: 'MiniGrid-MultiRoom-N2-S4-v0',
            4: 'MiniGrid-MultiRoom-N4-S5-v0',
            6: 'MiniGrid-MultiRoom-N6-v0'
        }
        return env_names_dict[conf['num_rooms']]

    elif conf['env_task'] == 'MiniGrid-LavaGap':
        env_names_dict = {
            5: 'MiniGrid-LavaGapS5-v0',
            6: 'MiniGrid-LavaGapS6-v0',
            7: 'MiniGrid-LavaGapS7-v0',
        }
        return env_names_dict[conf['s_size']]

    elif conf['env_task'] == 'MiniGrid-LavaCrossing':
        env_names_dict = {
            1: 'MiniGrid-LavaCrossingS9N1-v0',
            2: 'MiniGrid-LavaCrossingS9N2-v0',
            3: 'MiniGrid-LavaCrossingS9N3-v0',
            5: 'MiniGrid-LavaCrossingS11N5-v0',
        }
        return env_names_dict[conf['n_size']]

    elif conf['env_task'] == 'MiniGrid-SimpleCrossing':
        env_names_dict = {
            1: 'MiniGrid-SimpleCrossingS9N1-v0',
            2: 'MiniGrid-SimpleCrossingS9N2-v0',
            3: 'MiniGrid-SimpleCrossingS9N3-v0',
            5: 'MiniGrid-SimpleCrossingS11N5-v0',
        }
        return env_names_dict[conf['n_size']]

    return conf['env_task']


def gen_wrapped_env(conf, verbose=False):
    env_name = get_env_name(conf)

    if 'agent_start_pos' in conf:
        env = gym.make(env_name, agent_start_pos=conf['agent_start_pos'])
    else:
        env = gym.make(env_name)

    if 'minigrid_max_steps' in conf:
        env.max_steps = conf['minigrid_max_steps']

    if conf.get('fix_reset_seed', False):
        env = FixResetSeedWrapper(env, conf.get('reset_seed', 0))

    if conf.get('goal_patch', False):
        env = GoalPatch(env)

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
            env = ImgNorm(env)
        else:
            env = FullyObsWrapper(env)  # Fully observable gridworld using a compact grid encoding

    else:
        if conf.get('rgb_image', False):
            env = RGBImgPartialObsWrapper(env, tile_size=conf['tile_size'])  # Fully observed RGB image
            env = ImgNorm(env)
            #env = PosEncoding(env)

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
