import torch
from rllr.env import make_vec_envs
from pyhocon.config_parser import ConfigFactory
from train_master_gan import gen_env_with_seed, VAEEncoder, VAEDecoder, VAE, init_params
from tqdm import trange
from time import time
import matplotlib.pyplot as plt


def play():
    config = ConfigFactory.parse_file('conf/minigrid_second_step_random_empty.hocon')
    vae_model = VAE((64, 64, 3))
    vae_state_dict = torch.load('vae.pt', map_location='cpu')
    vae_model.load_state_dict(vae_state_dict)

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(config, env_id),
        num_processes=1,
        device='cpu'
    )

    agent = torch.load(config['outputs.model'], map_location='cpu')

    fig1, ax1 = plt.subplots()
    for _ in trange(100):
        obs, done, episode_reward = env.reset(), False, 0
        rnn_hxs = torch.zeros((1, config.get('encoder.recurrent_hidden_size', 1) * 2))
        masks = torch.ones((1, 1))

        while not done:
            value, action, _, rnn_hxs = agent.act(obs, rnn_hxs, masks, deterministic=False)
            obs, reward, done, infos = env.step(action)
            env.render('human')
            img = vae_model.decode(action)
            ax1.imshow(img[0].permute(1, 2, 0).numpy())
            time.sleep(1)
            episode_reward += float(reward)

        input(f'> the end!, reward = {episode_reward}')


if __name__ == '__main__':
    play()
