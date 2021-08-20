from rllr.env.gym_minigrid_navigation import environments as minigrid_envs
from rllr.env.gym_minigrid_navigation.environments import RGBImgObsWrapper, PosObsWrapper
from pyhocon import ConfigFactory
from experiments.train_worker import get_goal_achieving_criterion
from rllr.env.gym_minigrid_navigation.environments import random_grid_goal_generator
from matplotlib import pyplot as plt
import numpy as np
import gym
from matplotlib import pyplot as plt
import torch
from collections import defaultdict
from tqdm import trange


def make_env():
    env  = gym.make('MiniGrid-Dynamic-Obstacles-8x8-v0')
    env = RGBImgObsWrapper(env, tile_size=4)
    env = PosObsWrapper(env)
    return env


def rnd_obs(env, seed):
    env.seed(seed)
    while True:
        env.reset()
        goal_pos = np.random.randint(1, 6, 2)
        goal_dir = np.random.randint(0, 4)
        env.unwrapped.agent_pos = goal_pos
        env.unwrapped.agent_dir = goal_dir
        yield env.observation(env.unwrapped.gen_obs())


def make_dataset(generator, total_size=100000):
    dataset = defaultdict(list)
    for i, state in zip(trange(total_size), generator):
        dataset[str(state['position'])].append(state['image'])
    return dataset



def dist(encoder, state, goal_state):
    with torch.no_grad():
        embeds = encoder((torch.from_numpy(np.stack([state, goal_state]))))
        return torch.nn.CosineSimilarity(dim=0, eps=1e-08)(embeds[0], embeds[1])
    # return torch.dist(embeds[0], embeds[1], 2).cpu().numpy()


if __name__ == '__main__':
    env = make_env()
    dataset = make_dataset(rnd_obs(make_env(), 123), 100)
    encoder = torch.load('artifacts/models/minigrid_state_distance_encoder.p')

    max_dist = 0
    max_first = None
    max_second = None
    max_key = None
    for key in dataset:
        if len(dataset[key]) < 1:
            continue
        for first, second in zip(dataset[key][:-1], dataset[key][1:]):
            d = max(dist(encoder, first, second), dist(encoder, second, first))
            if d > max_dist:
                max_dist = d
                max_first = first
                max_second = second
                max_key = key
    print('same fig max_dist', max_dist)
    fig, axis = plt.subplots(1, 2)
    axis[0].imshow(max_first)
    axis[1].imshow(max_second)
    plt.title(f'dist = {max_dist}, key = {max_key}')
    plt.show()

    min_dist = np.inf
    min_first, first_key = None, None
    min_second, second_key = None, None
    keys = list(dataset.keys())
    count_fails = 0
    total = 0
    for i in trange(len(keys)):
        first = dataset[keys[i]][0]
        for j in range(i + 1, len(keys)):
            for second in dataset[keys[j]]:
                total += 1
                d = min(dist(encoder, first, second), dist(encoder, second, first))
                if d < min_dist:
                    min_dist = d
                    min_first, first_key = first, keys[i]
                    min_second, second_key = second, keys[j]
                count_fails += d < 19

    print('diff fig min_dist', min_dist)
    print(f'dist < 1 count: {count_fails} / {total}')
    fig, axis = plt.subplots(1, 2)
    axis[0].imshow(min_first)
    axis[1].imshow(min_second)
    plt.title(f'dist = {min_dist}, first_key = {first_key}, second_key {second_key}')
    plt.show()

