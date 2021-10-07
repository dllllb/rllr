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


def pos_to_obs(env, seed):
    env.seed(seed)
    def f(pos):
        dir = np.random.randint(0, 4)
        env.unwrapped.agent_pos = pos
        env.unwrapped.agent_dir = dir
        return env.observation(env.unwrapped.gen_obs())
    return f


def gen_grid():
    from itertools import product
    xs = range(1, 7)
    ys = range(1, 7)
    pos = product(xs, ys)
    return pos


def gen_obs(env, pos, seed):
    obs = map(lambda p: pos_to_obs(env, seed)(p), pos)
    return enumerate(obs)

def make_dataset(generator, total_size=100000):
    dataset = defaultdict(list)
    for i, state in zip(trange(total_size), generator):
        dataset[str(state['position'])].append(state['image'])
    return dataset

# def dist(encoder, state, goal_state):
#     with torch.no_grad():
#         embeds = encoder((torch.from_numpy(np.stack([state, goal_state]))))
#         return torch.nn.CosineSimilarity(dim=0, eps=1e-08)(embeds[0], embeds[1])
#     # return torch.dist(embeds[0], embeds[1], 2).cpu().numpy(


def dist(ssim, state, goal_state):
    with torch.no_grad():
        s1 = torch.from_numpy(np.array([state]))
        s2 = torch.from_numpy(np.array([goal_state]))
        return 1 - ssim.similarity(s1, s2)


if __name__ == '__main__':

    encoder = torch.load('artifacts/models/minigrid_ssim.p')

    env = make_env()

    # import pandas as pd
    #
    # pos = list(gen_grid())
    # idx = range(len(pos))
    # df_map = pd.DataFrame(index=idx, columns=idx)
    # for i, obs_i in gen_obs(env, pos, 1):
    #     for j, obs_j in gen_obs(env, pos, 2):
    #         df_map.loc[i, j] = dist(encoder, obs_i['image'], obs_j['image']).item()

    # df_map.fillna(0, inplace=True)
    # import seaborn as sns
    # sns.heatmap(df_map, annot=False)
    # plt.show()


    dataset = make_dataset(rnd_obs(make_env(), 123), 100)


    max_dist = 0
    max_first = None
    max_second = None
    max_key = None
    total = 0
    fn = 0
    tp = 0
    thd = 0.5


    for key in dataset:
        if len(dataset[key]) < 1:
            continue
        for first, second in zip(dataset[key][:-1], dataset[key][1:]):
            d = dist(encoder, first, second)
            if d > thd:
                fn += 1
            else:
                tp +=1
            total += 1
            if d > max_dist:
                max_dist = d
                max_first = first
                max_second = second
                max_key = key
    print(f'true positives {tp/total}, false negatives {fn/total}')
    print('same fig max_dist', max_dist)
    # fig, axis = plt.subplots(1, 2)
    # axis[0].imshow(max_first)
    # axis[1].imshow(max_second)
    # plt.title(f'dist = {max_dist}, key = {max_key}')
    # plt.show()

    min_dist = np.inf
    min_first, first_key = None, None
    min_second, second_key = None, None
    keys = list(dataset.keys())
    count_fails = 0
    total = 0
    fp = 0
    tn = 0
    for i in trange(len(keys)):
        first = dataset[keys[i]][0]
        for j in range(i + 1, len(keys)):
            for second in dataset[keys[j]]:
                total += 1
                d = dist(encoder, second, first)
                if d <= thd:
                    fp += 1
                else:
                    tn +=1
                if d < min_dist:
                    min_dist = d
                    min_first, first_key = first, keys[i]
                    min_second, second_key = second, keys[j]
                count_fails += d < 1e-9
    print(f'true negatives {tn/total}, false positives {fp/total}')
    print('diff fig min_dist', min_dist)
    print(f'dist < 1 count: {count_fails} / {total}')
    # fig, axis = plt.subplots(1, 2)
    # axis[0].imshow(min_first)
    # axis[1].imshow(min_second)
    # plt.title(f'dist = {min_dist}, first_key = {first_key}, second_key {second_key}')
    # plt.show()

