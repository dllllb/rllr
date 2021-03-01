import gym
import logging
import numpy as np

from gym.wrappers import Monitor
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


class NavigationGoalWrapper(gym.Wrapper):
    def __init__(self, env):
        self.goal_pos = None
        self.goal_state = None
        super().__init__(env)

    def observation(self, observation):
        return self.env.observation(observation)

    def step(self, action):
        next_state, _, _, info = self.env.step(action)
        next_pos = self.env.unwrapped.agent_pos

        done = (next_pos == self.goal_pos).all() or (self.step_count >= self.max_steps)

        return next_state, 0, done, info


class RandomPosAndGoalWrapper(NavigationGoalWrapper):
    def __init__(self, env, verbose=False):
        self.grid_size = env.unwrapped.grid.encode().shape[0]
        self.verbose = verbose
        super().__init__(env)

    def reset(self):
        # Generate random goal state
        self.goal_pos = np.random.randint(1, self.grid_size - 2, 2)
        self.env.unwrapped.agent_pos = self.goal_pos
        self.goal_state = self.env.observation(self.env.unwrapped.gen_obs())

        # Set random initial state
        self.env.reset()
        init_pos = None
        while init_pos is None or (init_pos == self.goal_pos).all():
            init_pos = np.random.randint(1, self.grid_size - 2, 2)
        self.env.unwrapped.agent_pos = init_pos

        if self.verbose:
            logger.info(f"From {init_pos} to {self.goal_pos}")

        # Return initial state
        return self.env.observation(self.env.unwrapped.gen_obs())


class FromBufferGoalWrapper(NavigationGoalWrapper):
    def __init__(self, env, verbose=False):
        self.grid_size = env.unwrapped.grid.encode().shape[0]
        self.verbose = verbose
        super().__init__(env)

    def reset(self):
        pass  # TOBD


class SetRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_function):
        self.reward_function = reward_function
        self.pos_reward = hasattr(reward_function, 'is_pos_reward') and reward_function.is_pos_reward
        super().__init__(env)

    def step(self, action):
        cur_pos = self.env.unwrapped.agent_pos
        state = self.env.observation(self.env.unwrapped.gen_obs())

        next_state, _, done, info = self.env.step(action)
        next_pos = self.env.unwrapped.agent_pos

        if self.pos_reward:
            reward = self.reward_function(cur_pos, next_pos, self.goal_pos)
        else:
            reward = self.reward_function(state, next_state, self.goal_state)

        return next_state, reward, done, info


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
        env = ImgObsWrapper(RGBImgObsWrapper(env, tile_size=conf['tile_size']))  # Fully observed RGB image

    goal_type = conf.get('goal_type', 'random')
    if goal_type == 'random':
        env = RandomPosAndGoalWrapper(env, verbose=verbose)  # env with random goal and init states
    elif goal_type == 'buffer':
        env = FromBufferGoalWrapper(env, verbose=verbose)  # env with goal and init states from buffer
    else:
        raise AttributeError(f"unknown goal_type '{conf['goal_type']}'")

    env = SetRewardWrapper(env, reward_function)  # set reward function

    env = FullyRenderWrapper(env)  # removes the default visualization of the partially observable field of view.
    if conf.get('video_path', False):
        env = Monitor(env, conf['video_path'], force=True)

    env.action_size = action_size

    return env
