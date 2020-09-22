import gym
from gym import error, spaces, utils
import numpy as np

class RGBImgAndStateObsWrapper(gym.core.ObservationWrapper):

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width*tile_size, self.env.height*tile_size, 3),
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped

        rgb_img = env.render(
            mode='rgb_array',
            highlight=False,
            tile_size=self.tile_size
        )

        return {
            'agent_pos': obs['agent_pos'],
            'image': rgb_img,
            'direction': [obs['direction']]
        }