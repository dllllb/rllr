import gym
import logging
import numpy as np
import torch

from argparse import ArgumentParser
from collections import defaultdict
from tqdm import trange, tqdm

from rllr.env.gym_minigrid_navigation.environments import RGBImgObsWrapper, PosObsWrapper, RandomStartPointWrapper, FixResetSeedWrapper
from rllr.utils.logger import init_logger

logger = logging.getLogger(__name__)


def make_env(env_name, reset_seed=None):
    env = gym.make(env_name)
    env = RandomStartPointWrapper(env, {})
    env = RGBImgObsWrapper(env, tile_size=4)
    env = PosObsWrapper(env)
    if reset_seed is not None:
        env = FixResetSeedWrapper(env, seed=reset_seed)
    return env


def rnd_obs(env, seed):
    env.seed(seed)
    while True:
        yield env.reset()


def make_dataset(generator, total_size=100000):
    dataset = defaultdict(list)
    for i, state in zip(trange(total_size), generator):
        dataset[str(state['position'])].append(state['image'])
    return dataset


def dist(ssim, state, goal_state):
    with torch.no_grad():
        s1 = torch.from_numpy(np.array([state]))
        s2 = torch.from_numpy(np.array([goal_state]))
        return (1 - ssim.similarity(s1, s2)).item()


def main(args):
    encoder = torch.load(args.model, map_location='cpu')

    dataset = make_dataset(rnd_obs(make_env(args.env, args.reset_seed), args.seed), args.episodes)
    thd = args.thd

    # same state distances
    total, max_dist, fn, tp = 0, 0, 0, 0
    for key, value in dataset.items():
        if not value:
            continue
        for first, second in zip(value[:-1], value[1:]):
            d = dist(encoder, first, second)
            if d > thd:
                fn += 1
            else:
                tp += 1
            total += 1
            if d > max_dist:
                max_dist = d

    logger.info(f'true positives {tp / total :.3f}, false negatives {fn / total :.3f}')
    logger.info(f'same fig max_dist {max_dist :.3f}')

    # different state distances
    min_dist = np.inf
    total, count_fails, fp, tn = 0, 0, 0, 0
    for first_key, first_value in tqdm(dataset.items()):
        first = first_value[0]
        for second_key, second_value in dataset.items():
            if first_key <= second_key:
                continue

            for second in second_value:
                total += 1
                d = dist(encoder, second, first)
                if d <= thd:
                    fp += 1
                else:
                    tn += 1
                if d < min_dist:
                    min_dist = d
                count_fails += d < 1e-9

    logger.info(f'true negatives {tn / total :.3f}, false positives {fp / total :.3f}')
    logger.info(f'diff fig min_dist: {min_dist :.3f}')
    logger.info(f'dist < 1 count: {count_fails} / {total}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--env', default='MiniGrid-Dynamic-Obstacles-8x8-v0', type=str)
    parser.add_argument('--model', default='../artifacts/minigrid/models/minigrid_ssim.p', type=str)
    parser.add_argument('--reset_seed', default=None, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--thd', default=0.5, type=float)
    parser.add_argument('--episodes', default=1000, type=int)
    args = parser.parse_args()

    init_logger(__name__)
    main(args)
