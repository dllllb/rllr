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
from torchvision.transforms import Normalize, ToTensor, Compose
from torch.utils.data import DataLoader
import torch.nn.functional as F


class BCEGANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, is_real):
        if is_real:
            target = torch.ones_like(x)
        else:
            target = torch.zeros_like(x)
        return self.loss(x, target)


class GAN(nn.Module):
    def __init__(self, generator, discriminator, conf):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.criterion = BCEGANLoss()
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(),
                                            lr=conf['training.generator_lr'],
                                            betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(),
                                            lr=conf['training.discriminator_lr'],
                                            betas=(0.5, 0.999))
        self.max_grad_norm = conf.get('training.max_grad_norm', 0.2)
        self.disc_update_n = conf.get('training.disc_update_n', 1)

    def gen_img(self, x):
        return self.generator(x)

    def disc_img(self, x):
        return self.discriminator(x)

    def forward(self, curr_state, real_next_state):
        return self.gen_img(curr_state)

    def update(self, curr_state, real_next_state):
        for _ in range(1):
            self.optimizer_D.zero_grad()
            with torch.no_grad():
                fake_next_step = self.gen_img(curr_state)
            fake_input = {'curr_state': curr_state,
                          'next_state': fake_next_step.detach()}
            real_input = {'curr_state': curr_state,
                          'next_state': real_next_state}

            fake_D = self.disc_img(fake_input)
            real_D = self.disc_img(real_input)

            #loss_D = torch.mean(fake_D) - torch.mean(real_D)
            loss_D_real = self.criterion(real_D, True)
            loss_D_fake = self.criterion(fake_D, False)
            loss_D = (loss_D_fake + loss_D_real) * 0.5

            loss_D.backward()
            self.optimizer_D.step()

            #for p in self.discriminator.parameters():
            #    p.data.clamp_(-0.01, 0.01)

        self.optimizer_G.zero_grad()
        fake_next_step = self.gen_img(curr_state)
        fake_input = {'curr_state': curr_state,
                      'next_state': fake_next_step}

        #loss_G = -torch.mean(self.disc_img(fake_input))
        loss_G = self.criterion(self.disc_img(fake_input), True)

        loss_G.backward()
        self.optimizer_G.step()

        return torch.mean(fake_D).detach().item(),\
               torch.mean(real_D).detach().item(),\
               loss_D.detach().item(),\
               loss_G.detach().item()


def get_gan(input_shape, conf):
    generator = encoders.get_encoder(None, conf['generator'], input_shape=input_shape)
    input_shape = {'curr_state': input_shape,
                   'next_state': input_shape}
    discriminator = encoders.get_encoder(None, conf['discriminator'], input_shape=input_shape)
    return GAN(generator, discriminator, conf)


def mnist_generator():
    tr = ToTensor()
    dataset = MNIST('/Users/al/prjs/temporal/rllr/', train=True, download=False, transform=tr)
    dl = DataLoader(dataset, batch_size=64, shuffle=True)
    while True:
        data = iter(dl)
        for x, _ in data:
            yield x, x


def gen_env(conf, seed):
    conf['env']['deterministic'] = True
    conf['env']['seed'] = seed
    verbose = 0
    env = environments.minigrid_envs.gen_wrapped_env(conf['env'], verbose=verbose)
    return EpisodeInfoWrapper(env)


def get_batch(env, batch_size=64):
    while True:
        batch = list()
        for _ in range(4):
            obs = env.reset()
            batch.append(obs)
        batch = torch.cat(batch, dim=0)
        batch = torch.permute(batch, (0, 3, 1, 2))
        yield batch


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
            state = torch.randn(1, 100)
            gen_state = gan.gen_img(state)[0]
            curr_state, next_state = curr_state[0], next_state[0]

            writer.add_image('curr state', curr_state, step)
            writer.add_image('gen next state', gen_state.reshape(1, 28, 28), step)
            torch.save(gan, conf['outputs.model'])


def main(args=None):
    conf = get_conf(args)
    switch_reproducibility_on(conf['seed'])
    input_shape = (1, 28, 28)
    gan = get_gan(input_shape, conf)
    gan.to(conf['training.device'])
    datagen = mnist_generator()
    train(gan, datagen, conf)


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    init_logger(__name__)
    init_logger('rllr.env.wrappers')
    main()
