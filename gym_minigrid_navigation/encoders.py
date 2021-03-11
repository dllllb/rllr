import torchvision.models as zoo_models

from functools import reduce
from torch import nn
import torch


class MLP(nn.Module):
    """Multi Layer Perceptron"""
    def __init__(self, input_size, conf):
        super().__init__()
        self.input_size = input_size

        layers = []
        layers_size = [input_size] + list(conf['hidden_layers_sizes'])
        for size_in, size_out in zip(layers_size[:-1], layers_size[1:]):
            layers.append(nn.Linear(size_in, size_out))
            layers.append(nn.ReLU())
        layers.append(nn.Tanh())
        self.output_size = layers_size[-1]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Flattener(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        if hasattr(model, 'output_size'):
            self.output_size = model.output_size

    def forward(self, x):
        x = x[:, 1:-1, 1:-1].flatten(start_dim=1)
        return self.model(x)


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.dims = args

    def forward(self, x):
        return x.permute(self.dims)


class SimpleCNN(nn.Module):
    def __init__(self, grid_size, conf):
        super().__init__()
        conv_layers = [Permute(0, 3, 1, 2)]
        cnn_output_size = grid_size
        cur_channels = 3
        for n_channels, kernel_size, max_pool in zip(conf['n_channels'], conf['kernel_sizes'], conf['max_pools']):
            conv_layers.append(nn.Conv2d(cur_channels, n_channels, kernel_size))
            cnn_output_size -= kernel_size - 1
            cur_channels = n_channels
            if max_pool > 1:
                conv_layers.append(nn.MaxPool2d(max_pool, max_pool))
                cnn_output_size //= max_pool

        self.conv_net = nn.Sequential(*conv_layers)
        self.output_size = cur_channels * cnn_output_size ** 2

    def forward(self, x):
        return self.conv_net(x).reshape(x.shape[0], -1)


class ResNet(nn.Module):
    def __init__(self, conf):
        super().__init__()
        if conf['layers_amount'] == 18:
            resnet = zoo_models.resnet18(pretrained=conf['pretrained'])
        elif conf['layers_amount'] == 34:
            resnet = zoo_models.resnet34(pretrained=conf['pretrained'])
        elif conf['layers_amount'] == 50:
            resnet = zoo_models.resnet50(pretrained=conf['pretrained'])
        elif conf['layers_amount'] == 101:
            resnet = zoo_models.resnet101(pretrained=conf['pretrained'])
        elif conf['layers_amount'] == 152:
            resnet = zoo_models.resnet152(pretrained=conf['pretrained'])
        else:
            raise AttributeError(f"Incorrect layers_amount: '{conf['layers_amount']}'. Could be 18/34/50/101/152")

        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # strips off last linear layer
        self.output_size = 512

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        return self.resnet(x).reshape(x.shape[0], -1)


def get_encoders(config):
    grid_size = config['env.grid_size'] * config['env'].get('tile_size', 1)

    # master
    if config['master.state_encoder_type'] == 'simple_mlp':
        state_size = 3 * (grid_size - 2) ** 2
        goal_state_encoder = Flattener(MLP(state_size, config['master']))
    elif config['master.state_encoder_type'] == 'simple_cnn':
        goal_state_encoder = SimpleCNN(grid_size, config['master'])
    elif config['master.state_encoder_type'] == 'resnet':
        goal_state_encoder = ResNet(config['master'])
    else:
        raise AttributeError(f"unknown master nn_type '{config['master.state_encoder_type']}'")

    # worker
    if config['worker.state_encoder_type'] == 'simple_mlp':
        state_size = 3 * (grid_size - 2) ** 2
        state_encoder = Flattener(MLP(state_size, config['worker']))
    elif config['worker.state_encoder_type'] == 'simple_cnn':
        state_encoder = SimpleCNN(grid_size, config['worker'])
    elif config['worker.state_encoder_type'] == 'resnet':
        state_encoder = ResNet(config['worker'])
    else:
        raise AttributeError(f"unknown worker nn_type '{config['worker.state_encoder_type']}'")

    return state_encoder, goal_state_encoder
