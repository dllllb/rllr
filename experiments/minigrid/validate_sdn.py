import gym
import logging
import numpy as np
import torch

from collections import defaultdict
from tqdm import trange

from rllr.env.gym_minigrid_navigation.environments import RGBImgObsWrapper, PosObsWrapper
from rllr.utils.logger import init_logger

logger = logging.getLogger(__name__)


def make_env():
    env = gym.make('MiniGrid-Dynamic-Obstacles-8x8-v0')
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


def dist(ssim, state, goal_state):
    with torch.no_grad():
        s1 = torch.from_numpy(np.array([state]))
        s2 = torch.from_numpy(np.array([goal_state]))
        return 1 - ssim.similarity(s1, s2)


def main():
    seed = 42
    encoder = torch.load('artifacts/models/minigrid_ssim.p')

    dataset = make_dataset(rnd_obs(make_env(), seed), 1000)

    max_dist = 0
    total = 0
    fn = 0
    tp = 0
    thd = 0.15

    for key in dataset:
        if len(dataset[key]) < 1:
            continue
        for first, second in zip(dataset[key][:-1], dataset[key][1:]):
            d = dist(encoder, first, second)
            if d > thd:
                fn += 1
            else:
                tp += 1
            total += 1
            if d > max_dist:
                max_dist = d

    logger.info(f'true positives {tp / total}, false negatives {fn / total}')
    logger.info('same fig max_dist', max_dist)

    min_dist = np.inf
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
                    tn += 1
                if d < min_dist:
                    min_dist = d

                count_fails += d < 1e-9
    logger.info(f'true negatives {tn / total}, false positives {fp / total}')
    logger.info('diff fig min_dist', min_dist.item())
    logger.info(f'dist < 1 count: {count_fails.item()} / {total}')


if __name__ == '__main__':
    init_logger(__name__)
    main()
