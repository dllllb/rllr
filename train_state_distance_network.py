import logging
import numpy as np
import os
import torch
import torch.nn as nn

from rllr.env.gym_minigrid_navigation.encoders import get_encoder
from rllr.env.gym_minigrid_navigation.environments import gen_wrapped_env
from rllr.models import StateDistanceNetwork
from rllr.utils import get_conf, init_logger, switch_reproducibility_on, convert_to_torch

logger = logging.getLogger(__name__)


def rollout(env, max_steps=False, device=torch.device("cpu")):
    states, next_states, actions = [], [], []
    done = False

    state = env.reset()
    i = 0

    while not done:
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        states.append(state)
        next_states.append(next_state)
        actions.append(action)
        state = next_state
        if i > max_steps and max_steps:
            break
        i += 1

    states = torch.from_numpy(np.array(states)).to(device, dtype=torch.float32)
    next_states = torch.from_numpy(np.array(next_states)).to(device, dtype=torch.float32)
    actions = torch.from_numpy(np.array(actions)).to(device)
    return states, next_states, actions


def main(args=None):
    config = get_conf(args)
    switch_reproducibility_on(config['seed'])

    env = gen_wrapped_env(config['env'])

    grid_size = config['env.grid_size'] * config['env'].get('tile_size', 1)
    encoder = get_encoder(grid_size, config['state_distance_encoder'])

    net = StateDistanceNetwork(encoder=encoder, action_size=env.action_space.n, config=config['state_distance_network'])

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-6)
    nll_loss = nn.NLLLoss()

    conf = config['training']
    device = torch.device(conf['device'])
    net.to(device)
    for roll in range(conf['n_episodes']):
        states, next_states, actions = rollout(env, device=device)

        loss = 0
        for epoch in range(conf['n_epochs']):
            optimizer.zero_grad()
            predicted_actions = net.forward(states, next_states)
            loss = nll_loss(predicted_actions, actions)
            loss.backward()
            optimizer.step()
        if roll % conf['verbose'] == 0:
            logger.info("Rollout: {0}, loss: {1}".format(roll, loss.item()))

    if config.get('outputs', False):
        if config.get('outputs.path', False):

            save_dir = os.path.dirname(config['outputs.path'])
            os.makedirs(save_dir, exist_ok=True)

            torch.save(net.encoder, config['outputs.path'])
            logger.info(f"Models saved to '{config['outputs.path']}'")

    if config['env.env_type'] == 'gym_minigrid':  # sanity check
        env.reset()
        states = []
        for action in [2, 0, 1]:
            state, _, _, _ = env.step(action)
            states.append(state)

        embeds = net.encoder(convert_to_torch(states).to(device))
        d_same = torch.dist(embeds[0], embeds[2], 2).detach().item()
        d_diff = torch.dist(embeds[0], embeds[1], 2).detach().item()

        logger.info("Sanity check:")
        logger.info(f"Same agent's position states distance: {d_same} (should be small)")
        logger.info(f"Different agent's position states distance: {d_diff} (should be bigger)")


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('gym_minigrid_navigation.environments')
    main()
