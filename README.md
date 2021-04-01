# Getting Started

## Variant 1: pipenv installation
```
sudo apt install python3.8 python3-venv
pip3 install pipenv

pipenv install --dev

pipenv shell

# install ffmpeg for videos (optional)
sudo apt install ffmpeg

# installing spunningup (optional)
cd ..
git clone https://github.com/openai/spinningup.git
cd spinningup
pip install -e .
cd rllr
```

## Variant 2: Conda Installation (deprecated)

### To install project environment

1. Create (and activate) a new environment with Python 3.6.
 
```bash
conda create --name rllr python=3.7
source activate rllr
```
    
2. Clone the repository and install dependencies.
```bash
git clone https://github.com/dllllb/rllr.git
git checkout dqn_rnd_goal
cd notebooks
pip install requirements.txt
```

3. Create an ipython kernel for the `rllr` environment.  
```bash
python -m ipykernel install --user --name rllr --display-name "Python 3.7 (rllr)"
```

## Pipeline
```bash
# minigrid environment
python train_state_distance_network.py --conf conf/minigrid_zero_step.hocon
python train_worker.py --conf conf/minigrid_first_step.hocon
python train_master.py --conf conf/minigrid_second_step.hocon
```
