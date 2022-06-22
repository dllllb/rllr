import torch

from env import gen_env_with_seed
from vae import VAE
from utils import train_vae
from matplotlib import pyplot as plt

from rllr.env import make_vec_envs


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(env_id),
        num_processes=32,
        device=device
    )

    vae = VAE(env.observation_space.shape, 32)
    train_vae(env, vae)
