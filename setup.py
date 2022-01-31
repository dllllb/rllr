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
    install_requires=[  # as declared im rllr's module code
        'numpy',
        'torch',
        'torchvision',
        'gym[accept-rom-license]',
        'ale-py',
        'matplotlib',
        'pyhocon',            # utils.config :: get_conf
        'jupyter',            # utils.plotting :: IPython
        'gym_minigrid',       # git+https://github.com/maximecb/gym-minigrid.git
        'scipy',              # environments.py :: scipy.stats
        'tqdm',
        'stable-baselines3',
        'opencv-python'
    ],
)

