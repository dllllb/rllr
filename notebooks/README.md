# Path rl experiments with minigrid

### Getting Started

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
