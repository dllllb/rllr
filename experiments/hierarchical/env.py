import gym
from gym_minigrid.wrappers import RGBImgObsWrapper
from rllr.env import EpisodeInfoWrapper
import numpy as np


class RandomGoalTestWrapper(gym.Wrapper):
    def __init__(self, env, tile_size):
        super().__init__(env)
        self.grid_size = env.unwrapped.grid.encode().shape[0]
        self.tile_size = tile_size
        self.goal = None

        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(0, 255, (3, tile_size * self.grid_size, tile_size * self.grid_size)),
            'goal': gym.spaces.Box(0, 255, (3, tile_size * self.grid_size, tile_size * self.grid_size)),
        })

        self.action_space = gym.spaces.Discrete(3)

    def reset(self):
        super().reset()

        agent_pos = self.env.unwrapped.agent_pos
        agent_dir = self.env.unwrapped.agent_dir

        self.env.unwrapped.agent_pos = np.random.randint(1, self.grid_size - 1, 2)
        self.env.unwrapped.agent_dir = np.random.randint(0, self.action_space.n)
        goal = env.render(
            mode='rgb_array',
            highlight=False,
            tile_size=self.tile_size
        )

        self.env.unwrapped.agent_pos = agent_pos
        self.env.unwrapped.agent_dir = agent_dir
        state = env.render(
            mode='rgb_array',
            highlight=False,
            tile_size=self.tile_size
        )
        print('here')

        return {'image': state, 'goal': goal}


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
    env = gym.make('MiniGrid-LavaCrossingS9N1-v0')
    env.seed(seed=seed)
    # env = RandomGoalWrapper(env, tile_size=8)
    env = RGBImgObsWrapper(env, tile_size=8)
    env = GoalWrapper(env)
    env = EpisodeInfoWrapper(env)
    return env


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    env = gen_env_with_seed(0)
    for _ in range(100):
        obs = env.reset()
        print(obs)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(obs['image'])
        ax[1].imshow(obs['goal'])
        plt.show()


