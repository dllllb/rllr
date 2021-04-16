from setuptools import setup

setup(
    name='rllr',
    version='0.1',
    description="""""",
    license='MIT',
    packages=[
        'rllr',
        'rllr.algo',  # essential RL learning algorithm
        'rllr.env',  # environments and wrappers
        'rllr.models',  # policies and models
        'rllr.buffer',  # replay buffer
        'rllr.utils',  # utilities
        'rllr.env.gym_minigrid_navigation',  # namespace only
    ],
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'gym[atari]',
        'matplotlib',
        'pyhocon',            # utils.py :: get_conf
        'gym_minigrid',       # git+https://github.com/maximecb/gym-minigrid.git
        'stable_baselines3',  # master_worker_*.py
        'scipy',              # environments.py :: scipy.stats
        'jupyter',            # utils.py :: IPython
    ],
)
