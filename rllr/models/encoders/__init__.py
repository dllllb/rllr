import torch
from torch import nn

from .common_encoders import CNNAllLayers, SimpleCNN, ResNet, Flattener, MLP, Permute
from .minigrid_encoders import DummyRawStateEncoder, DummyImageStateEncoder
from .vae import VAE
from .nle_encoders import NetHackEncoder


def get_encoder(grid_size, config):
    if config['state_encoder_type'] == 'simple_mlp':
        state_size = 3 * (grid_size - 2) ** 2
        state_encoder = Flattener(MLP(state_size, config['hidden_layers_sizes']))
    elif config['state_encoder_type'] == 'simple_cnn':
        state_encoder = SimpleCNN(grid_size, config)
    elif config['state_encoder_type'] == 'cnn_all_layers':
        state_encoder = CNNAllLayers(grid_size, config)
    elif config['state_encoder_type'] == 'resnet':
        state_encoder = ResNet(config)
    elif config['state_encoder_type'] == 'dummy_raw':
        state_encoder = DummyRawStateEncoder(grid_size)
    elif config['state_encoder_type'] == 'dummy_image':
        state_encoder = DummyImageStateEncoder(grid_size, config)
    elif config['state_encoder_type'] == 'net_hack_encoder':
        state_encoder = NetHackEncoder(**config)
    else:
        raise AttributeError(f"unknown nn_type '{config['master.state_encoder_type']}'")

    if config.get('use_image_wrapper', False):
        state_encoder = ImageEncoder(state_encoder)

    return state_encoder


class ImageEncoder(nn.Module):
    def __init__(self, state_encoder):
        super().__init__()
        self.state_encoder = state_encoder
        self.output_size = self.state_encoder.output_size

    def forward(self, x):
        return self.state_encoder(x['image'])


class GoalStateEncoder(nn.Module):
    def __init__(self, state_encoder, goal_state_encoder):
        super().__init__()
        self.state_encoder = state_encoder
        self.goal_state_encoder = goal_state_encoder
        self.output_size = state_encoder.output_size + goal_state_encoder.output_size

    def forward(self, states_and_goals):
        state_encoding = self.state_encoder.forward(states_and_goals['state'])

        if 'goal_emb' in states_and_goals.keys():
            goal_encoding = states_and_goals['goal_emb']
        elif states_and_goals['goal_state']:
            goal_encoding = self.goal_state_encoder.forward(states_and_goals['goal_state'])
        else:
            size = (*state_encoding.shape[:-1], self.goal_state_encoder.output_size)
            goal_encoding = torch.zeros(size=size, device=state_encoding.device)

        return torch.cat([state_encoding, goal_encoding], dim=-1)
