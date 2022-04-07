from vae import VAE
from utils import rollout
from matplotlib import pyplot as plt
from env import gen_env_with_seed
import torch
from rllr.env import make_vec_envs
from tqdm import tqdm
from sklearn.mixture import GaussianMixture


def fit_latent(env, vae):
    model = GaussianMixture(n_components=1)
    dataset = []
    for i in range(10):
        states = rollout(env)
        enc = vae.encode(states)
        dataset.append(enc)
    dataset = torch.cat(dataset, dim=0)
    print(dataset.shape)
    model.fit(dataset)
    return model


def test_reco(env, vae):
    states = rollout(env)
    reconstructions = vae.decode(vae.encode(states))

    sampler = fit_latent(env, vae)
    rnd_hid = torch.from_numpy(sampler.sample(states.shape[0])[0]).float()
    # rnd_hid = torch.randn((states.shape[0], 10))
    rnd_reconstructions = vae.decode(rnd_hid)

    ids = list(range(states.shape[0]))
    for i in tqdm(ids):
        img = states[i]
        rec = reconstructions[i]
        rnd_rec = rnd_reconstructions[i]

        f, axarr = plt.subplots(1, 3)
        axarr[0].imshow(img.permute(1, 2, 0) / 255.)
        axarr[0].set_title('source')
        axarr[1].imshow(rec.permute(1, 2, 0))
        axarr[1].set_title(f'reconstruction')
        axarr[2].imshow(rnd_rec.permute(1, 2, 0))
        axarr[2].set_title(f'rnd')
        plt.show()


if __name__ == '__main__':
    env = make_vec_envs(lambda env_id: lambda: gen_env_with_seed(env_id), 1, 'cpu')

    vae = VAE(env.observation_space.shape)
    vae.load_state_dict(torch.load('vae.pt', map_location='cpu'))
    test_reco(env, vae)