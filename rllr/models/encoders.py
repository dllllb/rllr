import torchvision.models as zoo_models

from torch import nn
import torch
import torch.nn.functional as F

from functools import reduce


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


class MLPEncoder(nn.Module):
    def __init__(self, conf):
        super().__init__()
        input_size = conf['input_size']
        hidden_layers_sizes = conf.get('hidden_layers_sizes', list())

        layers = list()
        for layer_size in hidden_layers_sizes:
            layers.append(nn.Linear(input_size, layer_size))
            layers.append(nn.ReLU())
            input_size = layer_size

        self.fc_net = nn.Sequential(*layers)
        self.output_size = input_size

    def forward(self, x):
        return self.fc_net(x.view(x.shape[0], -1))


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


class Split(nn.Module):
    def __init__(self, dim, channels_names=None, squeeze=False):
        super().__init__()
        self.dim = dim
        self.channels_names = channels_names
        self.squeeze = squeeze

    def forward(self, x):
        tensors = torch.split(x, 1, dim=self.dim)
        if self.squeeze:
            tensors = [t.squeeze(dim=self.dim) for t in tensors]

        if self.channels_names is not None:
            tensors = {name: tensor for name, tensor in zip(self.channels_names, tensors)}
        return tensors


class Concat(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if type(x) is list:
            return torch.cat(x, dim=self.dim)
        elif type(x) is dict:
            return torch.cat([v for v in x.values()], dim=self.dim)


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


class SimpleCNNEncoder(nn.Module):
    def __init__(self, grid_size, conf):
        super().__init__()
        conv_params = zip(
            conf['n_channels'],
            conf['kernel_sizes'],
            conf['max_pools'] if conf.get('max_pools', False) else [1] * len(conf['n_channels']),
            conf['strides'] if conf.get('strides', False) else [1] * len(conf['n_channels']),
            conf['paddings'] if conf.get('paddings', False) else [0] * len(conf['n_channels']),
        )
        pre_permute = conf.get('pre_permute', False)
        post_permute = conf.get('post_permute', False)
        input_channels = conf.get('input_channels', 3)
        self.unit_norm = conf.get('unit_norm', False)
        last_layer_activation = conf.get('last_layer_activation', True)

        if pre_permute:
            conv_layers = [Permute(*pre_permute)]
            test_input = torch.zeros(1, grid_size, grid_size, input_channels)
        else:
            conv_layers = list()
            test_input = torch.zeros(1, input_channels, grid_size, grid_size)
        cur_channels = input_channels
        for n_channels, kernel_size, max_pool, stride, padding in conv_params:
            conv_layers.append(nn.Conv2d(cur_channels, n_channels, kernel_size, stride, padding))
            conv_layers.append(nn.ReLU(inplace=True))
            cur_channels = n_channels
            if max_pool == 'global':
                conv_layers.append(nn.AdaptiveMaxPool2d(1))
            elif max_pool > 1:
                conv_layers.append(nn.MaxPool2d(max_pool, max_pool))

        if not last_layer_activation:
            conv_layers.pop(-1)
        if post_permute:
            conv_layers.append(Permute(*post_permute))

        self.conv_net = nn.Sequential(*conv_layers)

        with torch.no_grad():
            test_output = self.conv_net(test_input)
        self.output_shape = list(test_output.shape[1:])
        self.output_size = reduce((lambda x, y: x * y), self.output_shape)

    def forward(self, x):
        if self.unit_norm:
            x = self.conv_net(x.float() / 255.)
        else:
            x = self.conv_net(x.float())
        return x


class AttentionCNNNetwork(SimpleCNNEncoder):
    def __init__(self, grid_size, conf):
        super().__init__(grid_size, conf)
        conv_params = zip(
            conf['attention.n_channels'],
            conf['attention.kernel_sizes'],
            conf['attention.max_pools'] if conf.get('attention.max_pools', False) else [1] * len(conf['attention.n_channels']),
            conf['attention.strides'] if conf.get('attention.strides', False) else [1] * len(conf['attention.n_channels']),
            conf['attention.paddings'] if conf.get('attention.paddings', False) else [0] * len(conf['attention.n_channels']),
        )

        conv_layers = list()
        cur_channels = self.output_shape[0]
        for n_channels, kernel_size, max_pool, stride, padding in conv_params:
            conv_layers.append(nn.Conv2d(cur_channels, n_channels, kernel_size, stride, padding))
            conv_layers.append(nn.ReLU(inplace=True))
            cur_channels = n_channels
            if max_pool > 1:
                conv_layers.append(nn.MaxPool2d(max_pool, max_pool))
        conv_layers.append(nn.Conv2d(cur_channels, 1, 1, 1, 0))
        self.attention_net = nn.Sequential(*conv_layers)

        test_input = torch.zeros(1, *self.output_shape)
        with torch.no_grad():
            test_output = self.attention_net(test_input)

        hidden_layers_sizes = conf.get('hidden_layers_sizes', list())
        fc_layers = list()
        cur_size = self.output_shape[0]
        for layer_size in hidden_layers_sizes:
            fc_layers.append(nn.Linear(cur_size, layer_size))
            fc_layers.append(nn.ReLU(inplace=True))
            cur_size = layer_size
        self.fc_net = nn.Sequential(*fc_layers)

        self.output_size = cur_size

    def forward(self, x):
        x = super().forward(x)
        attention_map = torch.softmax(self.attention_net(x).view(x.shape[0], 1, -1), dim=-1).view(x.shape[0], 1, *x.shape[2:])
        x = (x * attention_map).sum(dim=(-1, -2))
        x = self.fc_net(x)
        return x


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


class OneHot(nn.Module):
    def __init__(self, n, *args):
        super().__init__()
        self.register_buffer('one_hot', torch.eye(n))

    def forward(self, x):
        return F.embedding(x, self.one_hot)


class MultiEmbeddingNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()

        embed_dims = conf['embed_dims']
        embed_n = conf['embed_n']
        self.embed_names = conf['embed_names']
        fc_layers = conf.get('fc_layers', [])

        embed_type = conf.get('embed_type', 'embed')
        if embed_type == 'embed':
            embed_l = nn.Embedding
        elif embed_type == 'ohe':
            embed_l = OneHot
        else:
            raise NotImplementedError(f'{embed_type} embeddings is not implemented')
        embed_layers = dict(zip(self.embed_names, [embed_l(n, dim) for n, dim in zip(embed_n, embed_dims)]))
        self.embed_layers = nn.ModuleDict(embed_layers)

        fc_in = sum(embed_dims)
        fc = list()
        for fc_out in fc_layers:
            fc.append(nn.Linear(fc_in, fc_out))
            fc.append(nn.ReLU())
            fc_in = fc_out
        self.fc_net = nn.Sequential(*fc)

        self.output_size = fc_in

    def forward(self, x):
        embeds = list()
        for n in self.embed_names:
            embeds.append(self.embed_layers[n](x[n].int()))
        embeds = torch.cat(embeds, dim=-1)
        return self.fc_net(embeds)


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
            goal_encoding = self.goal_state_encoder.forward(states_and_goals)
        state_encoding = self.state_encoder.forward(states_and_goals['image'])
        return torch.cat([state_encoding, goal_encoding], dim=1)


class ConvGoalStateEncoder(nn.Module):
    def __init__(self, goal_encoder, first_cnn_encoder, second_cnn_encoder, fc_encoder):
        super().__init__()
        self.goal_state_encoder = goal_encoder
        self.first_conv_encoder = first_cnn_encoder
        self.second_conv_encoder = second_cnn_encoder
        self.fc_state_encoder = fc_encoder
        self.output_size = self.fc_state_encoder.output_size

        self.conv_out_side = self.first_conv_encoder.output_shape[1]

    def forward(self, states_and_goals):
        if 'goal_emb' in states_and_goals.keys():
            goal_encoding = states_and_goals['goal_emb']
        else:
            goal_encoding = self.goal_state_encoder.forward(states_and_goals)
        goal_encoding = goal_encoding.view(*goal_encoding.shape, 1, 1)
        goal_encoding = goal_encoding.tile([1, 1, self.conv_out_side, self.conv_out_side])

        pre_encodings = self.first_conv_encoder(states_and_goals['image'])
        pre_encodings = torch.cat([pre_encodings, goal_encoding], dim=1)
        encodings = self.second_conv_encoder(pre_encodings).reshape(pre_encodings.shape[0], -1)
        encodings = self.fc_state_encoder(encodings)
        return encodings


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
    def __init__(self, model, output_size, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.model = model
        self.output_size = output_size
        self.rnn = nn.LSTM(model.output_size, output_size, num_layers=num_layers)

    def forward(self, out: torch.Tensor, rnn_rhs: torch.Tensor, masks: torch.Tensor):
        out = self.model(out)
        out, rnn_rhs = self._forward_rnn(out, rnn_rhs, masks)
        return out, rnn_rhs

    def flatten_hc(self, h, c):
        h = torch.permute(h, (1, 0, 2))
        h = h.reshape(h.size(0), -1)
        c = torch.permute(c, (1, 0, 2))
        c = c.reshape(h.size(0), -1)
        hxs = torch.cat([h, c], dim=1)
        return hxs

    def unflatten_hxs(self, n, hxs):
        hxs = hxs.view(n, 2, hxs.size(1) // 2)
        h = hxs[:, 0, :].unsqueeze(0)
        h = h.reshape(-1, getattr(self, 'num_layers', 1), h.size(-1) // getattr(self, 'num_layers', 1))
        h = torch.permute(h, (1, 0, 2))
        c = hxs[:, 1, :].unsqueeze(0)
        c = h.reshape(-1, getattr(self, 'num_layers', 1), c.size(-1) // getattr(self, 'num_layers', 1))
        c = torch.permute(c, (1, 0, 2))
        return h, c

    def _forward_rnn(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            n = x.size(0)
            h, c = self.unflatten_hxs(n, hxs)
            x, (h, c) = self.rnn(x.unsqueeze(0), (h * masks, c * masks))
            x = x.squeeze(0)
            hxs = self.flatten_hc(h, c)
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
            h, c = self.unflatten_hxs(N, hxs)
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
            hxs = self.flatten_hc(h, c)

        return x, hxs


class MultiNet(nn.Module):
    def __init__(self, grid_size, conf):
        super().__init__()
        self.models_names = conf['transformations']
        models = dict()

        for model_name in self.models_names:
            models[model_name] = get_encoder(grid_size, conf[model_name])

        self.models = nn.ModuleDict(models)
        self.output_size = None


class Sequence(MultiNet):
    def __init__(self, grid_size, conf):
        super().__init__(grid_size, conf)
        #self.output_size = self.models[self.models_names[-1]].output_size
        self.output_shape = [23, 10, 10]
        self.output_size = 32


    def forward(self, x):
        for model_name in self.models_names:
            x = self.models[model_name](x)
        return x


class Parallel(MultiNet):
    def __init__(self, grid_size, conf):
        super().__init__(grid_size, conf)
        self.output_size = 0
        for model_name in self.models_names:
            self.output_size += self.models[model_name].output_size

    def forward(self, x):
        outputs = list()
        for model_name in self.models_names:
            outputs.append(self.models[model_name](x[model_name]))
        return torch.cat(outputs, dim=-1)


class ImageTextEncoder(nn.Module):
    def __init__(self, image_shape, sent_shape):
        super(ImageTextEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(image_shape[2], 16, kernel_size=(4, 4), stride=(4, 4)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Flatten()
        )

        self.word_enc = nn.Linear(in_features=sent_shape[1], out_features=sent_shape[1], bias=False)
        self.rnn = nn.LSTM(sent_shape[1], 128, batch_first=True)

        self.output_size = 576 + 128

    def forward(self, t):
        img, msg = t['state'].float() / 255., t['mission']
        img_enc = self.cnn(img.permute(0, 3, 1, 2))

        word_enc = self.word_enc(msg.view(-1, msg.shape[2])).view(msg.shape[0], msg.shape[1], msg.shape[2])
        out, (h, c) = self.rnn(word_enc)
        msg_enc = h.squeeze(dim=0)

        return torch.cat([img_enc, msg_enc], dim=1)


def get_encoder(grid_size, config):
    if config['state_encoder_type'] == 'simple_mlp':
        state_size = 3 * (grid_size - 2) ** 2
        state_encoder = Flattener(MLP(state_size, config['hidden_layers_sizes']))
    elif config['state_encoder_type'] == 'mlp_encoder':
        state_encoder = MLPEncoder(config)
    elif config['state_encoder_type'] == 'simple_cnn':
        state_encoder = SimpleCNN(grid_size, config)
    elif config['state_encoder_type'] == 'cnn_encoder':
        state_encoder = SimpleCNNEncoder(grid_size, config)
    elif config['state_encoder_type'] == 'attention_cnn_encoder':
        state_encoder = AttentionCNNNetwork(grid_size, config)
    elif config['state_encoder_type'] == 'cnn_all_layers':
        state_encoder = CNNAllLayers(grid_size, config)
    elif config['state_encoder_type'] == 'resnet':
        state_encoder = ResNet(config)
    elif config['state_encoder_type'] == 'dummy_raw':
        state_encoder = DummyRawStateEncoder(grid_size)
    elif config['state_encoder_type'] == 'dummy_image':
        state_encoder = DummyImageStateEncoder(grid_size, config)
    elif config['state_encoder_type'] == 'cnn_rnn':
        cnn_encoder = SimpleCNN(grid_size, config)
        state_encoder = RNNEncoder(cnn_encoder,
                                   config.get('rnn_output', cnn_encoder.output_size),
                                   config.get('rnn_num_layers', 1))
    elif config['state_encoder_type'] == 'multi_embeddings':
        state_encoder = MultiEmbeddingNetwork(config)
    elif config['state_encoder_type'] == 'cnn_hindsight_encoder':
        goal_encoder = get_encoder(grid_size, config['goal_encoder'])
        first_cnn_encoder = get_encoder(grid_size, config['first_cnn_encoder'])
        config['second_cnn_encoder']['input_channels'] = goal_encoder.output_size + first_cnn_encoder.output_shape[0]
        grid_size = first_cnn_encoder.output_shape[1]
        second_cnn_encoder = get_encoder(grid_size, config['second_cnn_encoder'])
        grid_size = second_cnn_encoder.output_size
        config['fc_encoder']['input_size'] = second_cnn_encoder.output_size
        fc_encoder = get_encoder(grid_size, config['fc_encoder'])
        state_encoder = ConvGoalStateEncoder(goal_encoder, first_cnn_encoder, second_cnn_encoder, fc_encoder)
    elif config['state_encoder_type'] == 'sequence':
        state_encoder = Sequence(grid_size, config)
    elif config['state_encoder_type'] == 'parallel':
        state_encoder = Parallel(grid_size, config)
    elif config['state_encoder_type'] == 'permute':
        state_encoder = Permute(*config['channels'])
    elif config['state_encoder_type'] == 'split':
        state_encoder = Split(dim=config['dim'],
                              channels_names=config.get('channels_names', None),
                              squeeze=config.get('squeeze', False))
    elif config['state_encoder_type'] == 'concat':
        state_encoder = Concat(dim=config['dim'])
    else:
        raise AttributeError(f"unknown nn_type '{config['master.state_encoder_type']}'")

    if config.get('normalize', False):
        state_encoder = NormEncoder(state_encoder)

    return state_encoder
