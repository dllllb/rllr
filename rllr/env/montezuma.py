import numpy as np
import cv2
import time

import gym
from gym.wrappers import TimeLimit


class StickyActionEnv(gym.Wrapper):
    def __init__(self, env, p=0.25):
        super(StickyActionEnv, self).__init__(env)
        self.p = p
        self.last_action = 0

    def step(self, action):
        if np.random.uniform() < self.p:
            action = self.last_action

        self.last_action = action
        return self.env.step(action)

    def reset(self):
        self.last_action = 0
        return self.env.reset()


class RepeatActionEnv(gym.Wrapper):
    def __init__(self, env, render):
        gym.Wrapper.__init__(self, env)
        self.successive_frame = np.zeros((2,) + self.env.observation_space.shape, dtype=np.uint8)
        self.render = render

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        reward, done = 0, False
        for t in range(4):
            state, r, done, info = self.env.step(action)
            if self.render:
                self.env.render()
                time.sleep(0.01)

            if t == 2:
                self.successive_frame[0] = state
            elif t == 3:
                self.successive_frame[1] = state
            reward += r
            if done:
                break

        state = self.successive_frame.max(axis=0)
        return state, reward, done, info


class MontezumaInfoWrapper(gym.Wrapper):
    def __init__(self, env):
        super(MontezumaInfoWrapper, self).__init__(env)
        self.room_address = 3
        self.visited_rooms = set()
        self.observation_space = gym.spaces.Box(0, 255, (1, 84, 84), dtype=np.uint8)

        self.episode_reward = 0
        self.episode_steps = 0

    def get_current_room(self):
        ram = self.unwrapped.ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        action = int(action)

        obs, rew, done, info = self.env.step(action)
        self.episode_reward += rew
        self.episode_steps += 1
        self.visited_rooms.add(self.get_current_room())

        if done:
            info['episode'] = {
                'r': self.episode_reward,
                'steps': self.episode_steps,
                'task': 'Montezuma',
                'visited_rooms':  len(self.visited_rooms)
            }
            self.visited_rooms.clear()

        return self.observation(obs), np.sign(rew), done, info

    def reset(self):
        self.episode_reward = 0
        self.episode_steps = 0
        return self.observation(self.env.reset())

    def observation(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        return np.stack([img])


def gen_env_with_seed(seed, render=False):
    env = gym.make('MontezumaRevengeNoFrameskip-v4')
    env = StickyActionEnv(env)
    env = RepeatActionEnv(env, render)
    env = MontezumaInfoWrapper(env)
    env = TimeLimit(env, max_episode_steps=4500)
    env.seed(seed)
    return env
