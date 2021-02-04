# PathRL pre-training algorithm

https://www.overleaf.com/read/snttfkctdvbd


## Getting Started

### Variant 1: pipenv installation
```
sudo apt install python3.8 python3-venv
pip3 install pipenv

pipenv install --dev

pipenv shell
```

### Variant 2: Conda Installation

#### To install project environment

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

## Master-Worker Step
```bash
# example:
# simple mlp
python navigation_policy.py --conf gym_minigrid_navigation/conf/dqn_navigation_mlp.hocon

# simple cnn
python navigation_policy.py \
    master.state_encoder_type="simple_cnn" \
    worker.state_encoder_type="simple_cnn" \
    outputs.path="outputs/models/dqn_simple_cnn.p" \
    --conf gym_minigrid_navigation/conf/dqn_navigation_mlp.hocon

# resnet on RGB picture
python navigation_policy.py --conf gym_minigrid_navigation/conf/dqn_navigation_resnet.hocon
```
