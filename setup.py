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
        'rllr.utils.plotting',  # visualization
        'rllr.env.gym_minigrid_navigation',  # namespace only
        'rllr.exploration',
    ],
    install_requires=[
    ],
)
