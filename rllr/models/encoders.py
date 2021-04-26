import torch
import torchvision.models as zoo_models

from torch import nn
import torch


class MLP(nn.Module):
    """Multi Layer Perceptron"""
    def __init__(self, input_size: int, hidden_layers_sizes: list):
        super().__init__()
        self.input_size = input_size

        layers = []
        layers_size = [input_size] + list(hidden_layers_sizes)
        for size_in, size_out in zip(layers_size[:-1], layers_size[1:]):
            layers.append(nn.Linear(size_in, size_out))
            layers.append(nn.ReLU())
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
        conv_params = zip(
            conf['n_channels'],
            conf['kernel_sizes'],
            conf['max_pools'] if conf.get('max_pools', False) else [1] * len(conf['n_channels']),
            conf['strides'] if conf.get('strides', False) else [1] * len(conf['n_channels'])
        )
        for n_channels, kernel_size, max_pool, stride in conv_params:
            conv_layers.append(nn.Conv2d(cur_channels, n_channels, kernel_size, stride))
            conv_layers.append(nn.ReLU(inplace=True))
            cnn_output_size += -1 * kernel_size + stride
            cnn_output_size //= stride
            cur_channels = n_channels
            if max_pool > 1:
                conv_layers.append(nn.MaxPool2d(max_pool, max_pool))
                cnn_output_size //= max_pool
        self.conv_net = nn.Sequential(*conv_layers)
        output_size = cur_channels * cnn_output_size ** 2

        hidden_layers_sizes = list(conf.get('hidden_layers_sizes', []))
        if hidden_layers_sizes:
            fc_layers = []
            layers_size = [output_size] + hidden_layers_sizes
            for size_in, size_out in zip(layers_size[:-1], layers_size[1:]):
                fc_layers.append(nn.Linear(size_in, size_out))
                fc_layers.append(nn.ReLU())
            self.fc = nn.Sequential(*fc_layers)
            output_size = layers_size[-1]
        else:
            self.fc = None

        self.output_size = output_size

    def forward(self, x):
        res = self.conv_net(x).reshape(x.shape[0], -1)
        if self.fc is not None:
            res = self.fc(res)
        return res


class CNNAllLayers(nn.Module):
    def __init__(self, grid_size, conf):
        super().__init__()
        self.transform = Permute(0, 3, 1, 2)
        self.conv_layers = []
        cnn_output_size = grid_size
        cur_channels = 3
        conv_params = zip(
            conf['n_channels'],
            conf['kernel_sizes'],
            conf['max_pools'] if conf.get('max_pools', False) else [1] * len(conf['n_channels']),
            conf['strides'] if conf.get('strides', False) else [1] * len(conf['n_channels'])
        )
        self.output_size = 0
        for n_channels, kernel_size, max_pool, stride in conv_params:
            conv = nn.Conv2d(cur_channels, n_channels, kernel_size, stride)
            cnn_output_size += -1 * kernel_size + stride
            cnn_output_size //= stride
            cur_channels = n_channels
            if max_pool > 1:
                layer = nn.Sequential(conv, nn.ReLU(inplace=True), nn.MaxPool2d(max_pool, max_pool))
                cnn_output_size //= max_pool
            else:
                layer = nn.Sequential(conv, nn.ReLU(inplace=True))

            self.conv_layers.append(layer)
            self.output_size += cur_channels * cnn_output_size ** 2
        self.nn = nn.Sequential(*self.conv_layers)

    def forward(self, x):
        x = self.transform(x)
        result = []
        for layer in self.conv_layers:
            x = layer(x)
            result.append(x.reshape(x.shape[0], -1))

        return torch.cat(result, dim=1)


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


class DummyRawStateEncoder(nn.Module):
    """
    code minigrid raw states as agent position and direction via one hot encoder
    """
    def __init__(self, grid_size):
        super().__init__()
        self.output_size = (grid_size - 2) ** 2 + 4

    def forward(self, x: torch.Tensor):
        x = x[:, 1:-1, 1:-1]
        agent_pos = (x[:, :, :, 0] == 10)
        directions = x[agent_pos][:, -1:]

        agent_pos = agent_pos.flatten(start_dim=1).float()
        directions = torch.eq(directions, torch.arange(4, device=x.device)).float()

        return torch.cat([agent_pos, directions], dim=1)


class DummyImageStateEncoder(nn.Module):
    """
    code minigrid image states as agent position and direction via one hot encoder
    """
    def __init__(self, grid_size, conf):
        super().__init__()
        self.tile_size = conf['tile_size']
        self.output_size = (grid_size // self.tile_size - 2) ** 2 + 4

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 1, 2)
        x = x[:, :1, self.tile_size:-self.tile_size, self.tile_size:-self.tile_size]
        a = nn.MaxPool2d(self.tile_size, self.tile_size)(x) > 0

        b = x[a.repeat_interleave(self.tile_size, dim=-2).repeat_interleave(self.tile_size, dim=-1)]
        b = b.reshape(-1, self.tile_size, self.tile_size)
        b = [b[:, :, -1].sum(dim=1) == 0, b[:, -1].sum(dim=1) == 0, b[:, :, 0].sum(dim=1) == 0, b[:, 0].sum(dim=1) == 0]

        directions = torch.cat([x.unsqueeze(-1) for x in b], dim=1).float()
        agent_pos = a.flatten(start_dim=1).float()

        return torch.cat([agent_pos, directions], dim=1)


class GoalStateEncoder(nn.Module):

    def __init__(self, state_encoder, goal_state_encoder):
        super().__init__()
        self.state_encoder = state_encoder
        self.goal_state_encoder = goal_state_encoder
        self.output_size = state_encoder.output_size + goal_state_encoder.output_size

    def forward(self, states_and_goals):
        if 'goal_emb' in states_and_goals.keys():
            goal_encoding = states_and_goals['goal_emb']
        else:
            goal_encoding = self.goal_state_encoder.forward(states_and_goals['goal_state'])
        state_encoding = self.state_encoder.forward(states_and_goals['state'])
        return torch.cat([state_encoding, goal_encoding], dim=1)


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
    else:
        raise AttributeError(f"unknown nn_type '{config['master.state_encoder_type']}'")

    return state_encoder
