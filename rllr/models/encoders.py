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
            conf['strides'] if conf.get('strides', False) else [1] * len(conf['n_channels']),
            conf['paddings'] if conf.get('paddings', False) else [0] * len(conf['n_channels']),
        )
        for n_channels, kernel_size, max_pool, stride, padding in conv_params:
            conv_layers.append(nn.Conv2d(cur_channels, n_channels, kernel_size, stride, padding))
            conv_layers.append(nn.ReLU(inplace=True))
            cnn_output_size += -1 * kernel_size + stride + 2 * padding
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

        directions = torch.eq(directions, torch.arange(4, device=x.device)).float()

        return torch.cat([x[:, :, :, 0].flatten(start_dim=1), directions], dim=1)


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


class RemoveOuterWalls(nn.Module):
    def __init__(self, model, tile_size):
        super().__init__()
        self.tile_size = tile_size
        self.model = model
        if hasattr(model, 'output_size'):
            self.output_size = model.output_size

    def forward(self, x):
        return self.model(x[:, self.tile_size: -self.tile_size, self.tile_size: -self.tile_size])


class NormEncoder(nn.Module):
    def __init__(self, model, eps=1e-9):
        super().__init__()
        self.eps = eps
        self.model = model
        self.output_size = model.output_size

    def forward(self, out: torch.Tensor):
        out = self.model(out)
        return out / (out.pow(2).sum(dim=1) + self.eps).pow(0.5).unsqueeze(-1).expand(*out.size())


class LastActionEncoder(nn.Module):
    def __init__(self, enc, n_actions):
        super().__init__()
        self.state_enc = enc
        self.act_enc = nn.Embedding(n_actions, embedding_dim=16)
        self.output_size = enc.output_size + 16

    def forward(self, t, *args):
        state_enc, rnn_hxs = self.state_enc(t['state'], *args)
        act_enc = self.act_enc(t['last_action'].long())
        return torch.cat([state_enc, act_enc], dim=1), rnn_hxs


class RNNEncoder(nn.Module):
    def __init__(self, model, output_size):
        super().__init__()
        self.model = model
        self.output_size = output_size
        self.rnn = nn.LSTM(model.output_size, output_size)

    def forward(self, out: torch.Tensor, rnn_rhs: torch.Tensor, masks: torch.Tensor):
        out = self.model(out)
        out, rnn_rhs = self._forward_rnn(out, rnn_rhs, masks)
        return out, rnn_rhs

    def _forward_rnn(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            hxs = hxs.view(x.size(0), 2, hxs.size(1) // 2)
            h = hxs[:, 0, :]
            c = hxs[:, 1, :]
            x, (h, c) = self.rnn(x.unsqueeze(0), ((h * masks).unsqueeze(0), (c * masks).unsqueeze(0)))
            x = x.squeeze(0)
            hxs = torch.cat([h.squeeze(0), c.squeeze(0)], dim=1)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.view(N, 2, -1)
            h = hxs[:, 0, :].unsqueeze(0)
            c = hxs[:, 1, :].unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, (h, c) = self.rnn(
                    x[start_idx:end_idx],
                    (h * masks[start_idx].view(1, -1, 1), c * masks[start_idx].view(1, -1, 1)),
                )

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = torch.cat([h.squeeze(0), c.squeeze(0)], dim=1)

        return x, hxs


# Todo: finish sigmoid sampling
def gumbel_sigmoid(logits, tau):
    r"""Straight-through gumbel-sigmoid estimator
    """
    gumbels = -torch.empty_like(logits,
                                memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.sigmoid()

    # Straight through.
    y_hard = (y_soft > 0.5).long()
    ret = y_hard - y_soft.detach() + y_soft

    return ret


class ImageTextEncoder(nn.Module):
    def __init__(self, image_shape, sent_shape):
        super(ImageTextEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(image_shape[2], 16, kernel_size=(4, 4), stride=(4, 4)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            #nn.Flatten()
        )

        self.word_enc = nn.Linear(in_features=sent_shape[1], out_features=sent_shape[1], bias=False)
        self.rnn = nn.LSTM(sent_shape[1], 128, batch_first=True)
        self.film = nn.Linear(128, 16)

        self.output_size = 576 + 576

        self.train()

    def forward(self, t):
        img, msg = t['state'].float() / 255., t['mission']
        img_enc = self.cnn(img.permute(0, 3, 1, 2))

        word_enc = self.word_enc(msg.view(-1, msg.shape[2])).view(msg.shape[0], msg.shape[1], msg.shape[2])
        out, (h, c) = self.rnn(word_enc)
        msg_enc = h.squeeze(dim=0)
        chan_mask = self.sample_channels_mask(logits=self.film(msg_enc)).unsqueeze(dim=2).unsqueeze(dim=3)
        return torch.cat([img_enc, img_enc * chan_mask], dim=1)\.view(img.shape[0], -1)

    def sample_channels_mask(self, logits):
        """
            Samples binary mask to select
            relevant output channel of the convolution
            Attributes:
            logits - logprobabilities of the bernoully variables
                for each output channel of the convolution to be selected
        """
        if self.training:
            channels_mask = gumbel_sigmoid(logits, tau=2 / 3)
        else:
            channels_mask = (logits > 0).long()
        return channels_mask


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
