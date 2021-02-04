import gym
import logging
import numpy as np

from gym.wrappers import Monitor
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLORS
from gym_minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper

logger = logging.getLogger(__name__)


class FullyRenderWrapper(gym.Wrapper):
    def render(self, *args, **kwargs):
        """This removes the default visualization of the partially observable field of view."""
        kwargs['highlight'] = False
        return self.unwrapped.render(*args, **kwargs)


class ImgObsWrapper(gym.Wrapper):
    """
    Use the image as the only observation output, no language/mission.
    """
    def observation(self, obs):
        return self.env.observation(obs)['image']

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return next_state['image'], reward, done, info


class RandomPosAndGoalWrapper(gym.Wrapper):
    def __init__(self, env, reward_function, conf, verbose=False):
        self.goal_state = None
        self.grid_size = env.unwrapped.grid.encode().shape[0]
        self.reward_function = reward_function
        self.rgb_image = conf.get('rgb_image', False)
        self.verbose = verbose
        super().__init__(env)

    def reset(self):
        # Generate goal state
        goal_pos = np.random.randint(1, self.grid_size - 2, 2)
        self.env.unwrapped.agent_pos = goal_pos
        self.goal_state = self.env.observation(self.env.unwrapped.gen_obs())

        # Set initial state
        self.env.reset()
        init_pos = None
        while init_pos is None or (init_pos == goal_pos).all():
            init_pos = np.random.randint(1, self.grid_size - 2, 2)

        self.env.unwrapped.agent_pos = init_pos

        if self.verbose:
            logger.info(f"From {init_pos} to {goal_pos}")

        # Return initial state
        return self.env.observation(self.env.unwrapped.gen_obs())

    def step(self, action):
        state = self.env.observation(self.env.unwrapped.gen_obs())
        next_state, reward, done, info = self.env.step(action)
        cur_pos, next_pos, goal_pos = self.agent_pos_coords(state), self.agent_pos_coords(next_state), self.agent_pos_coords(self.goal_state)

        reward = self.reward_function(cur_pos, next_pos, goal_pos)

        if (next_pos == goal_pos).all() or (self.step_count >= self.max_steps):
            done = True
        else:
            done = False

        return next_state, reward, done, info

    def agent_pos_coords(self, state):
        if not self.rgb_image:
            obj_pos = (state[:, :, 0].T == OBJECT_TO_IDX['agent']).nonzero()
            return np.array(next(zip(*obj_pos)))
        else:
            # TODO: agent != red colour
            color = COLORS['red']  # agent = red colour

            obj_pos = (state == color).all(axis=2).nonzero()
            return np.array(next(zip(*obj_pos))) // 8


def gen_wrapped_env(conf, reward_function, verbose=False):
    if conf['env_task'] == 'MiniGrid-Empty':
        env_name = f"{conf['env_task']}-{conf['grid_size']}x{conf['grid_size']}-v0"
        action_size = 3
    else:
        raise AttributeError(f"unknown env_task '{conf['env_task']}'")

    env = gym.make(env_name)
    if not conf.get('rgb_image', False):
        env = ImgObsWrapper(FullyObsWrapper(env))  # Fully observable gridworld using a compact grid encoding
    else:
        env = ImgObsWrapper(RGBImgObsWrapper(env))  # Fully observed RGB image

    env = RandomPosAndGoalWrapper(env, reward_function, conf, verbose=verbose)

    env = FullyRenderWrapper(env)  # removes the default visualization of the partially observable field of view.
    if conf.get('video_path', False):
        env = Monitor(env, conf['video_path'], force=True)

    env.action_size = action_size

    return env
