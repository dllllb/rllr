import gym
from gym_minigrid.wrappers import RGBImgObsWrapper
from rllr.env import EpisodeInfoWrapper


class GoalWrapper(gym.Wrapper):
    def __init__(self, env):
        super(GoalWrapper, self).__init__(env)
        img_shape = env.observation_space['image'].shape
        self.observation_space = gym.spaces.Box(0, 255, (img_shape[2], img_shape[0], img_shape[1]))
        self.action_space = gym.spaces.Discrete(3)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info

    def reset(self, **kwargs):
        return self.observation(self.env.reset())

    def observation(self, obs):
        return obs['image'].transpose(2, 0, 1)


def gen_env_with_seed(seed):
    env = gym.make('MiniGrid-Empty-8x8-v0', agent_start_pos=None)
    env.seed(seed=seed)
    env = RGBImgObsWrapper(env, tile_size=8)
    env = GoalWrapper(env)
    env = EpisodeInfoWrapper(env)
    return env

