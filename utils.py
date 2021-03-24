import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_conf(args=None):
    import sys
    import os
    from pyhocon import ConfigFactory

    p = argparse.ArgumentParser()
    p.add_argument('-c', '--conf', nargs='+')
    args, overrides = p.parse_known_args(args)

    logger.info(f'args: {args}, overrides: {overrides}')

    init_conf = f"script_path={os.path.dirname(os.path.abspath(sys.argv[0]))}"
    file_conf = ConfigFactory.parse_string(init_conf)

    if args is not None and args.conf is not None:
        for name in args.conf:
            logger.info(f'Load config from "{name}"')
            file_conf = ConfigFactory.parse_file(name, resolve=False).with_fallback(file_conf, resolve=False)

    overrides = ','.join(overrides)
    over_conf = ConfigFactory.parse_string(overrides)
    if len(over_conf) > 0:
        logger.info(f'New overrides:')

        def print_differences(root=''):
            if len(root) > 0:
                c = over_conf[root[:-1]]
            else:
                c = over_conf

            for k, v in c.items():
                old = file_conf.get(f"{root}{k}", None)
                if isinstance(v, dict) and isinstance(old, dict):
                    print_differences(f'{root}{k}.')
                else:
                    logger.info(f'    For key "{root}{k}" provided new value "{v}", was "{old}"')

        print_differences()
    conf = over_conf.with_fallback(file_conf)
    return conf


def init_logger(name, level=logging.INFO, filename=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)-20s   : %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename, mode='w')
        formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)-20s   : %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def display_stats(scores, steps):
    fig = plt.figure()
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Max Score')
    plt.xlabel('Episode #')
    plt.show()

    fig = plt.figure()
    plt.plot(np.arange(len(steps)), steps)
    plt.ylabel('Steps')
    plt.xlabel('Episode #')
    plt.show()


def switch_reproducibility_on(seed=42):
    import torch
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def convert_to_torch(arr):
    if arr and isinstance(arr[0], dict):
        arr = [x['image'] for x in arr]
    arr = np.vstack([np.expand_dims(x, axis=0) for x in arr])
    return torch.from_numpy(arr).float()
