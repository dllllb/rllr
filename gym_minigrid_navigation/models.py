import torchvision.models as zoo_models

from functools import reduce
from torch import nn


class MLP(nn.Module):
    """Multi Layer Perceptron"""
    def __init__(self, input_size, hidden_layers_sizes=(64, 64)):
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


def compose(*fs):
    """Functions composition"""
    def _compose2(f, g):
        return lambda *a, **kw: f(g(*a, **kw))

    return reduce(_compose2, fs)


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.dims = args

    def forward(self, x):
        return x.permute(self.dims)


class SimpleCNN(nn.Module):
    def __init__(self, grid_size, n_channels1=6, n_channels2=16):
        super().__init__()
        conv_layers = [
            (Permute(0, 3, 1, 2), lambda x: x),  # channels first
            (nn.Conv2d(3, n_channels1, 3), lambda x: x - 2),
            (nn.MaxPool2d(2, 2), lambda x: x // 2),
            (nn.Conv2d(n_channels1, n_channels2, 3), lambda x: x - 2)
        ]
        self.conv_net = nn.Sequential(*[x[0] for x in conv_layers])
        cnn_output_size = compose(*[x[1] for x in conv_layers])(grid_size)
        self.output_size = n_channels2 * cnn_output_size ** 2

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
    grid_size = config['env.grid_size']

    # master
    if config['master.state_encoder_type'] == 'simple_mlp':
        state_size = 3 * (grid_size - 2) ** 2
        goal_state_encoder = Flattener(MLP(state_size))
    elif config['master.state_encoder_type'] == 'simple_cnn':
        goal_state_encoder = SimpleCNN(grid_size=grid_size)
    elif config['master.state_encoder_type'] == 'resnet':
        goal_state_encoder = ResNet(config['master'])
    else:
        raise AttributeError(f"unknown master nn_type '{config['master.state_encoder_type']}'")

    # worker
    if config['worker.state_encoder_type'] == 'simple_mlp':
        state_size = 3 * (grid_size - 2) ** 2
        state_encoder = Flattener(MLP(state_size))
    elif config['worker.state_encoder_type'] == 'simple_cnn':
        state_encoder = SimpleCNN(grid_size=grid_size)
    elif config['worker.state_encoder_type'] == 'resnet':
        state_encoder = ResNet(config['worker'])
    else:
        raise AttributeError(f"unknown worker nn_type '{config['worker.state_encoder_type']}'")

    return state_encoder, goal_state_encoder
