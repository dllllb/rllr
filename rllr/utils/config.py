import os
import sys
import logging
import argparse

from pyhocon import ConfigFactory

logger = logging.getLogger(__name__)


def get_conf(args=None):
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
