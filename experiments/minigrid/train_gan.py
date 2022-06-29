# import some shit
import logging
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import rllr.env as environments
import rllr.models as models

from rllr.env import make_vec_envs, minigrid_envs, EpisodeInfoWrapper
from rllr.models import encoders
from rllr.utils import train_ppo, im_train_ppo, get_conf, switch_reproducibility_on
from rllr.utils.logger import init_logger

from tqdm import trange
import numpy as np
import random

from torchvision.datasets import MNIST
from torchvision.transforms import PILToTensor
from torch.utils.data import DataLoader


class ImagePool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images


class GAN(nn.Module):
    def __init__(self, generator, discriminator, conf):
        super().__init__()
        self.image_pool = ImagePool(conf.get('training.pool_size', 1000))
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=conf['training.generator_lr'])
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=conf['training.discriminator_lr'])
        self.max_grad_norm = conf.get('training.max_grad_norm', 0.2)
        self.disc_update_n = conf.get('training.disc_update_n', 1)

    def gen_img(self, x):
        return (self.generator(x) + 1) / 2 * 255

    def disc_img(self, x):
        return self.discriminator(x)

    def forward(self, curr_state, real_next_state):
        return self.gen_img(curr_state)

    def update(self, curr_state, real_next_state):
        for _ in range(self.disc_update_n):
            self.optimizer_D.zero_grad()
            with torch.no_grad():
                fake_next_step = self.gen_img(curr_state)
            fake_input = {'curr_state': curr_state,
                          'next_state': fake_next_step.detach()}
            real_input = {'curr_state': curr_state,
                          'next_state': real_next_state}

            fake_D = self.disc_img(fake_input)
            real_D = self.disc_img(real_input)
            loss_D = torch.mean(fake_D - real_D)
            loss_D.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        fake_next_step = self.gen_img(curr_state)
        fake_input = {'curr_state': curr_state,
                      'next_state': fake_next_step}
        loss_G = -torch.mean(self.disc_img(fake_input))
        loss_G.backward()
        nn.utils.clip_grad_norm_(self.generator.parameters(), self.max_grad_norm)
        self.optimizer_G.step()

        return torch.mean(fake_D).detach().item(),\
               torch.mean(real_D).detach().item(),\
               loss_D.detach().item(),\
               loss_G.detach().item()


# gen env
def gen_env(conf, seed):
    conf['env']['deterministic'] = True
    conf['env']['seed'] = seed
    verbose = 0
    env = environments.minigrid_envs.gen_wrapped_env(conf['env'], verbose=verbose)
    return EpisodeInfoWrapper(env)


# make gan
def get_gan(input_shape, conf):
    generator = encoders.get_encoder(None, conf['generator'], input_shape=input_shape)
    input_shape = {'curr_state': input_shape,
                   'next_state': input_shape}
    discriminator = encoders.get_encoder(None, conf['discriminator'], input_shape=input_shape)
    return GAN(generator, discriminator, conf)


def get_random_agent(conf):
    action_size = conf['env.action_size']
    n_processes = conf['training.n_processes']
    def agent(*args, **kwargs):
        return np.random.randint(0, action_size, n_processes)
    return agent


def random_walk_generator(env, agent, conf):
    agent_steps = conf['agent_steps']
    n_processes = conf['n_processes']
    batch_size = conf['batch_size']
    per_batch_episodes = int(batch_size/n_processes)

    while True:
        batch_curr_state, batch_next_state = list(), list()
        for _ in range(per_batch_episodes):
            curr_state = env.reset()
            next_state = curr_state
            for _ in range(agent_steps):
                actions = torch.Tensor(agent(next_state)).int()
                next_state, _, _, _ = env.step(actions)
            batch_curr_state.append(curr_state)
            batch_next_state.append(next_state)
        yield torch.cat(batch_curr_state, dim=0), torch.cat(batch_next_state, dim=0)


def mnist_generator():
    dataset = MNIST('/Users/al/prjs/temporal/rllr/', train=True, download=False, transform=PILToTensor())
    dl = DataLoader(dataset, batch_size=16, shuffle=True)
    while True:
        data = iter(dl)
        for x, _ in data:
            yield x, x


# train gan
def train(gan, data_gen, conf):
    writer = SummaryWriter(conf['outputs.logs'])
    n_steps = conf['training.n_steps']
    device = conf['training.device']
    verbose = conf['training.verbose']
    for step in trange(conf['training.n_steps']):
        curr_state, next_state = next(data_gen)
        curr_state = curr_state.to(device)
        next_state = next_state.to(device)

        loss_D_fake, loss_D_real, loss_D, loss_G = gan.update(curr_state, next_state)
        writer.add_scalar('loss_D_fake', loss_D_fake, step)
        writer.add_scalar('loss_D_real', loss_D_real, step)
        writer.add_scalar('loss_D', loss_D, step)
        writer.add_scalar('loss_G', loss_G, step)

        if step % verbose == 0:
            curr_state, next_state = curr_state[0], next_state[0]
            gen_state = gan.gen_img(curr_state.unsqueeze(0))[0]

            #.permute(2, 0, 1)
            writer.add_image('curr state', (curr_state / 255.).permute(2, 0, 1), step)
            writer.add_image('real next state', (next_state / 255.).permute(2, 0, 1), step)
            writer.add_image('gen next state', (gen_state / 255.).permute(2, 0, 1), step)
            torch.save(gan, conf['outputs.model'])


# main
def main(args=None):
    conf = get_conf(args)
    switch_reproducibility_on(conf['seed'])

    env = make_vec_envs(lambda env_id: lambda: gen_env(conf, env_id),
                        conf['training.n_processes'],
                        conf['training.device'])

    input_shape = env.observation_space.shape
    #input_shape = (1, 28, 28)

    gan = get_gan(input_shape, conf)
    gan.to(conf['training.device'])

    datagen = random_walk_generator(env, get_random_agent(conf), conf['training'])
    #datagen = mnist_generator()

    train(gan, datagen, conf)


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    init_logger(__name__)
    init_logger('rllr.env.wrappers')
    main()
