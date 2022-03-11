import numpy as np
from tqdm import trange, tqdm
import torch
from matplotlib import pyplot as plt


def rollout(env):
    obs, done, info = env.reset(), False, []

    states = []
    for _ in range(128):
        states.append(obs)
        action = torch.randint(0, 3, (obs.size(0), 1))
        obs, reward, done, info = env.step(action)
    states.append(obs)

    print(torch.stack(states, dim=0).transpose(1, 0).shape)
    return torch.stack(states, dim=0).transpose(1, 0).reshape(-1, *env.observation_space.shape)


def train_vae(env, master_agent, n_epoch=1000, batch_size=32):
    for epoch in trange(n_epoch):
        states = rollout(env)
        rec_loss = master_agent.update_vae(states, batch_size=batch_size)
        print('rec_loss', rec_loss)
        torch.save(master_agent.actor_critic.state_dict(), 'master_agent.pt')


def test_vae(env, master):
    states = rollout(env)
    rnd_ids = torch.randint(0, states.shape[0] - 1, (states.shape[0],))
    ids = list(range(states.shape[0]))
    for i in tqdm(ids):
        img = states[i].unsqueeze(dim=0)
        img_2 = states[i + 1].unsqueeze(dim=0)
        img_rnd = states[i + 7].unsqueeze(dim=0)
        from torch.nn import functional as F

        def sim(first, second):
            return F.cosine_similarity(
                master.actor_critic.vae.encode(first),
                master.actor_critic.vae.encode(second),
            ).item()

        f, axarr = plt.subplots(1, 3)
        axarr[0].imshow(img[0].permute(1, 2, 0))
        axarr[0].set_title('source')
        axarr[1].imshow(img_2[0].permute(1, 2, 0))
        axarr[1].set_title(f'next_step  {sim(img, img_2):.3f}')
        axarr[2].imshow(img_rnd[0].permute(1, 2, 0))
        axarr[2].set_title(f'random  {sim(img, img_rnd):.3f}')
        plt.show()
        continue

        with torch.no_grad():
            z_rnd = torch.randn((1, 256))
            rec_rnd = master.actor_critic.vae.decode(z_rnd)

            # z_mast = master.act(img)[1]
            z_mast = master.actor_critic.vae.encode(img)
            rec_img = master.actor_critic.vae.decode(z_mast)
        f, axarr = plt.subplots(1, 3)
        axarr[0].imshow(img[0].permute(1, 2, 0))
        axarr[0].set_title('source')
        axarr[1].imshow(rec_img[0].permute(1, 2, 0))
        axarr[1].set_title('reconstruction')
        axarr[2].imshow(rec_rnd[0].permute(1, 2, 0))
        axarr[2].set_title('random')
        plt.show()

