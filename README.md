# RLLR -- High-level Reinforcement Control

## Getting Started

### Install Git LFS

```brew install git-lfs```  
```git lfs install```

### Variant 1: pipenv installation

Pipenv is a tool that creates and manages a virtualenv for your projects, and track packages
from your Pipfile as you install/uninstall packages. It also allows to produce deterministic
builds.

```
sudo apt install python3.8 python3-venv
pip3 install pipenv

# install ffmpeg for videos (optional)
sudo apt install ffmpeg

# Editable developemnt install
cd rllr
pipenv install --dev -e .

# activate pipenv shell
pipenv shell

# installing spunningup (optional)
cd ..
git clone https://github.com/openai/spinningup.git
cd spinningup
pip install -e .
cd rllr
```

### Variant 2: Conda Install

Below we provide the essential steps to install via `conda`.

```bash
# create a `pysandbox` env for the lib
conda create -n pysandbox "python>=3.8" pip cython numpy mkl numba scipy scikit-learn \
jupyter ipython pytest "conda-forge::compilers>=1.0.4" conda-forge::llvm-openmp \
matplotlib pytorch::pytorch pytorch::torchvision pytorch::torchaudio "cudatoolkit>=10.2"

conda activate pysandbox

# [optional] jpyter kernel for the environment
# python -m ipykernel install --user --name pysandbox --display-name "Py3.8 (rllr)"

# clone and install the repo
git clone https://github.com/dllllb/rllr.git
cd rllr
pip install -e .  # editable installation accessible form anywhere

# run unit tests to validate installation
pytest tests/
```

## Pipeline
Please, refer to `./experimetns` for more detailed description of the experiments.

```bash
# minigrid environment
cd experiments

python train_state_distance_network.py --conf conf/minigrid_zero_step.hocon

python train_worker.py --conf conf/minigrid_first_step.hocon

python train_master.py --conf conf/minigrid_second_step.hocon

python enjoy.py --mode master


```
## Benchmarks 
| task / method                            | PPO         | PPO + RND | PATH RL |
| ---------------------------------------- | ----------- | ----------| --------|
| MiniGrid-Dynamic-Obstacles-8x8-v0        |             |  0.61     | 0.90    |
| MiniGrid-LavaCrossingS9N3-v0             |             |  0.90     |         |
| MiniGrid-DoorKey-8x8-v0                  |             |  0.91     |         |
| MiniGrid-FourRooms-v0                    |             |  0.72     |         |
| MiniGrid-KeyCorridorS3R3-v0              |             |  0.71     |         |
| MiniGrid-PutNear-6x6-N2-v0               |             |  0.78     |         |

<p align="center">
    Table 1. MiniGrid, fully observed: Mean return over 1000 episodes
</p> 


| task / method  | PPO         | PPO + RND | PATH RL |
| ---------------------------------------- | ----------- | ---------| ---------|
| MiniGrid-Dynamic-Obstacles-8x8-v0        |             | 110.77   | 25.32    |
| MiniGrid-LavaCrossingS9N3-v0             |             | 35.45    |          |
| MiniGrid-DoorKey-8x8-v0                  |             | 66.11    |          |
| MiniGrid-FourRooms-v0                    |             | 30.85    |          |
| MiniGrid-KeyCorridorS3R3-v0              |             | 88.18    |          |
| MiniGrid-PutNear-6x6-N2-v0               |             | 7.25     |          |

<p align="center">
    Table 2. MiniGrid, fully observed: Mean number of steps over 1000 episodes
</p> 


| task / method  | PPO         | PPO + RND | PATH RL |
| ---------------------------------------- | ----------- | ----------| --------|
| MiniGrid-Dynamic-Obstacles-8x8-v0        |             | 0.97      | 0.99    |
| MiniGrid-LavaCrossingS9N3-v0             |             | 1.00      |         |
| MiniGrid-DoorKey-8x8-v0                  |             | 1.00      |         |
| MiniGrid-FourRooms-v0                    |             | 0.98      |         |
| MiniGrid-KeyCorridorS3R3-v0              |             | 0.99      |         |
| MiniGrid-PutNear-6x6-N2-v0               |             | 0.99      |         |

<p align="center">
    Table 3. MiniGrid, fully observed: Mean success rate over 1000 episodes
</p> 