import gym
import numpy as np

from gym.wrappers import Monitor
from gym_minigrid.wrappers import FullyObsWrapper


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
    def __init__(self, env, reward_function):
        self.goal_state = None
        self.grid_size = env.unwrapped.grid.encode().shape[0]
        self.reward_function = reward_function
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

        # Return initial state
        return self.env.observation(self.env.unwrapped.gen_obs())

    def to_coords(self, state, idx=10):
        obj_pos = (state[:, :, 0] == idx).nonzero()
        return np.array(next(zip(*obj_pos)))

    def step(self, action):
        state = self.env.observation(self.env.unwrapped.gen_obs())
        next_state, reward, done, info = self.env.step(action)
        cur_pos, next_pos, goal_pos = self.to_coords(state), self.to_coords(next_state), self.to_coords(self.goal_state)

        reward = self.reward_function(cur_pos, next_pos, goal_pos)

        if (next_pos == goal_pos).all() or (self.step_count >= self.max_steps):
            done = True
        else:
            done = False

        return next_state, reward, done, info


def gen_wrapped_env(env_name, reward_function):
    env = gym.make(env_name)
    env = FullyObsWrapper(env)  # Fully observable gridworld using a compact grid encoding
    env = ImgObsWrapper(env)  # Use the image as the only observation output, no language/mission.
    env = RandomPosAndGoalWrapper(env, reward_function)
    env = FullyRenderWrapper(env)  # removes the default visualization of the partially observable field of view.
    env = Monitor(env, './video', force=True)
    return env
