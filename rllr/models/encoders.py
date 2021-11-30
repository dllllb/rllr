import torch
import torchvision.models as zoo_models
import torch.nn.functional as F
from torch import nn

from functools import reduce

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
        res = self.conv_net(x.float() / 255.).reshape(x.shape[0], -1)
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


class NormEncoder(nn.Module):
    def __init__(self, model, eps=1e-9):
        super().__init__()
        self.eps = eps
        self.model = model
        self.output_size = model.output_size

    def forward(self, out: torch.Tensor):
        out = self.model(out)
        return out / (out.pow(2).sum(dim=1) + self.eps).pow(0.5).unsqueeze(-1).expand(*out.size())


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

    if config.get('normalize', False):
        state_encoder = NormEncoder(state_encoder)

    return state_encoder


class VAE(nn.Module):
    def __init__(self, input_size, n_channels, kernel_sizes, strides, hidden_sizes):
        super(VAE, self).__init__()

        # Encoder convolutional layers
        conv_layers_enc = [Permute(0, 3, 1, 2)]
        self.input_size = input_size
        cnn_output_size = self.input_size[:-1]
        cur_channels = self.input_size[-1]
        conv_params = zip(n_channels, kernel_sizes, strides)
        for n_channel, kernel_size, stride in conv_params:
            conv_layers_enc.append(nn.Conv2d(cur_channels, n_channel, kernel_size, stride))
            conv_layers_enc.append(nn.ReLU(inplace=True))
            cnn_output_size_x = (cnn_output_size[0] - kernel_size + stride) // stride
            cnn_output_size_y = (cnn_output_size[1] - kernel_size + stride) // stride
            cnn_output_size = [cnn_output_size_x, cnn_output_size_y]
            cur_channels = n_channel
        self.enc_conv_net = nn.Sequential(*conv_layers_enc)
        self.cnn_output_sizes = [cur_channels] + cnn_output_size

        # Encoder fully connected layers
        fc_layers_enc = []
        output_size = reduce(lambda x, y: x * y, self.cnn_output_sizes)
        for hidden_size in hidden_sizes:
            fc_layers_enc.append(nn.Linear(output_size, hidden_size))
            fc_layers_enc.append(nn.ReLU(inplace=True))
            output_size = hidden_size

        fc_layers = fc_layers_enc[:-1]
        self.enc_mu = nn.Sequential(*fc_layers)
        self.enc_var = nn.Sequential(*fc_layers)

        # Decoder fully connected layers
        fc_layes_dec = []
        input_size = hidden_sizes[-1]
        output_size = reduce(lambda x, y: x * y, self.cnn_output_sizes)
        for hidden_size in hidden_sizes[:-1][::-1]:
            fc_layes_dec.append(nn.Linear(input_size, hidden_size))
            fc_layes_dec.append(nn.ReLU(inplace=True))
            input_size = hidden_size
        fc_layes_dec.append(nn.Linear(input_size, output_size))
        fc_layes_dec.append(nn.ReLU(inplace=True))
        self.dec_fc = nn.Sequential(*fc_layes_dec)
        conv_layers_dec = []

        # Decoder deconvolution layers
        n_channels = [3] + n_channels[:-1]
        deconv_params = zip(n_channels[::-1], kernel_sizes[::-1], strides[::-1])
        for n_channel, kernel_size, stride in deconv_params:
            conv_layers_dec.append(nn.ConvTranspose2d(cur_channels, n_channel, kernel_size, stride))
            conv_layers_dec.append(nn.ReLU(inplace=True))
            cur_channels = n_channel
        conv_layers_dec = conv_layers_dec[:-1]
        conv_layers_dec.append(Permute(0, 2, 3, 1))
        self.dec_conv_net = nn.Sequential(*conv_layers_dec)

    def encoder(self, x):
        x = self.enc_conv_net(x)
        x = x.view(-1, reduce(lambda x, y: x * y, self.cnn_output_sizes))
        mu = self.enc_mu(x)
        log_var = self.enc_var(x)
        return mu, log_var

    def _reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):
        x = self.dec_fc(z)
        x = x.view(-1, *self.cnn_output_sizes)
        x = self.dec_conv_net(x)
        return torch.sigmoid(x)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self._reparameterize(mu, log_var)
        out = self.decoder(z)
        return out, mu, log_var


class VarEmbedding(nn.Module):
    def __init__(self, embedding_size, state_encoder):
        super().__init__()
        self.state_encoder = state_encoder
        input_size = state_encoder.output_size


        self.mu = nn.Sequential(nn.Linear(input_size, embedding_size))
        self.logstd = nn.Sequential(nn.Linear(input_size, embedding_size))
        self.output_size = embedding_size

    def forward(self, states):
        states_encoding = self.state_encoder(states)
        mu = self.mu(states_encoding)
        logstd = self.logstd(states_encoding)
        sigma = logstd.exp()

        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)
        return z

    def kl_loss(self, states):
        states_encoding = self.state_encoder(states)
        mu = self.mu(states_encoding)
        logstd = self.logstd(states_encoding)
        KLD = -0.5 * torch.sum(1 + 2 * logstd - mu.pow(2) - (2 * logstd).exp())
        return KLD
