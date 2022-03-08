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

    return torch.cat(states, dim=0)


def train_vae(env, master_agent, n_epoch=1000, batch_size=32):
    for epoch in trange(n_epoch):
        states = rollout(env)
        rec_loss = master_agent.update_vae(states, batch_size=batch_size)
        print('rec_loss', rec_loss)
        torch.save(master_agent.actor_critic.state_dict(), 'master_agent.pt')


def test_vae(env, vae):
    states = rollout(env)
    ids = torch.randint(0, states.shape[0] - 1, (states.shape[0],))
    for i in tqdm(ids):
        img = states[i].unsqueeze(dim=0)

        with torch.no_grad():
            rec = vae.decode(vae.encode(img))
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(img[0].permute(1, 2, 0))
        axarr[1].imshow(rec[0].permute(1, 2, 0))
        plt.show()

