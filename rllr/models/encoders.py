import torchvision.models as zoo_models

from torch import nn
import torch
import torch.nn.functional as F

from functools import reduce, partial
from rllr.utils.common import get_output_shape


#LEGACY
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
    def __init__(self, input_shape, conf):
        super().__init__()
        input_size = input_shape[-1]
        hidden_layers_sizes = conf.get('hidden_layers_sizes', list())
        drop_out_rates = conf.get('dropout', [0 for _ in hidden_layers_sizes])
        use_batch_norm = conf.get('batch_norm', [0 for _ in hidden_layers_sizes])
        last_layer_activation = conf.get('last_layer_activation', True)
        self.unit_norm = conf.get('unit_norm', False)

        activation_type = conf.get('activation_type', 'relu')
        if activation_type == 'relu':
            activation = partial(nn.ReLU, inplace=True)
        elif activation_type == 'leaky_relu':
            activation = partial(nn.LeakyReLU, 0.2, inplace=True)
        else:
            raise NotImplementedError('Activation function not implemented')

        layers = list()
        for layer_size, drop_out_rate, batch_norm in zip(hidden_layers_sizes[:-1], drop_out_rates[:-1], use_batch_norm[:-1]):
            layers.append(nn.Linear(input_size, layer_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(layer_size))
            layers.append(activation())
            if drop_out_rate > 0:
                layers.append(nn.Dropout(drop_out_rate))
            input_size = layer_size

        layers.append(nn.Linear(input_size, hidden_layers_sizes[-1]))
        if last_layer_activation:
            layers.append(activation())
        if drop_out_rates[-1] > 0:
            layers.append(nn.Dropout(drop_out_rates[-1]))

        self.fc_net = nn.Sequential(*layers)
        self.output_shape = (*input_shape[:-1], input_size)

    def forward(self, x):
        if self.unit_norm:
            x = x.float() / 255.
        return self.fc_net(x)


class Sigmoid(nn.Module):
    def __init__(self, input_shape, conf):
        super().__init__()
        self.output_shape = input_shape

    def forward(self, x):
        return torch.sigmoid(x)


class Tanh():
    def __init__(self, input_shape, config):
        super().__init__()
        self.output_shape = input_shape

    def forward(self, x):
        return torch.tanh(x)


#LEGACY
class Flattener(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        if hasattr(model, 'output_size'):
            self.output_size = model.output_size

    def forward(self, x):
        x = x[:, 1:-1, 1:-1].flatten(start_dim=1)
        return self.model(x)


class Flatten(nn.Module):
    def __init__(self, input_shape, conf):
        super().__init__()
        self.start_dim = conf.get('start_dim', 1)
        self.output_shape = (reduce((lambda x, y: x * y), input_shape[self.start_dim-1:]),)

    def forward(self, x):
        return x.flatten(start_dim=self.start_dim)


class Reshape(nn.Module):
    def __init__(self, input_shape, conf):
        super().__init__()
        self.output_shape = conf['shape']

    def forward(self, x):
        return x.reshape(x.shape[0], *self.output_shape)


class Permute(nn.Module):
    def __init__(self, input_shape, *args):
        super().__init__()
        input_shape = ['bs', *input_shape]
        self.output_shape = tuple([input_shape[d] for d in args])[1:]
        self.dims = args

    def forward(self, x):
        return x.permute(self.dims)


class Split(nn.Module):
    def __init__(self, input_shape, config):
        super().__init__()
        self.dim = config['dim'],
        self.channels_names = config.get('channels_names', None),
        self.squeeze = config.get('squeeze', False)

        base_shape = list(input_shape)
        if self.squeeze:
            base_shape.pop(self.dim)
        else:
            base_shape[self.dim] = 1
        base_shape = tuple(base_shape)

        if self.channels_names is not None:
            self.output_shape = {name: base_shape for name in self.channels_names}
        else:
            self.output_shape = {i: base_shape for i in range(input_shape[self.dim])}

    def forward(self, x):
        tensors = torch.split(x, 1, dim=self.dim)
        if self.squeeze:
            tensors = [t.squeeze(dim=self.dim) for t in tensors]

        if self.channels_names is not None:
            tensors = {name: tensor for name, tensor in zip(self.channels_names, tensors)}
        else:
            tensors = {i: tensor for i, tensor in enumerate(tensors)}
        return tensors


class Concat(nn.Module):
    def __init__(self, input_shape, config):
        super().__init__()
        self.dim = config['dim']

        assert isinstance(input_shape, dict), f'input shape for Concat layer must be dict'
        base_output_shape = list(list(input_shape.values())[0])
        concat_dim_size = sum([shape[self.dim] for shape in input_shape.values()])
        base_output_shape[self.dim] = concat_dim_size
        self.output_shape = tuple(base_output_shape)

    def forward(self, x):
        if type(x) is list:
            return torch.cat(x, dim=self.dim)
        elif type(x) is dict:
            return torch.cat([x[k] for k in sorted(x.keys())], dim=self.dim)


class DictChoose(nn.Module):
    def __init__(self, input_shape, conf):
        super().__init__()
        self.key = conf['key']

        assert isinstance(input_shape, dict), f'input shape for Concat layer must be dict'
        self.output_shape = input_shape[self.key]

    def forward(self, x):
        return x[self.key]


class NoiseVec(nn.Module):
    def __init__(self, input_shape, conf):
        super().__init__()
        self.vec_size = conf['vec_size']
        self.output_shape = (self.vec_size,)

    def forward(self, x):
        return torch.randn((x.shape[0], self.vec_size))


#LEGACY
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
    def __init__(self, input_shape, conf):
        super().__init__()
        conv_params = zip(
            conf['n_channels'],
            conf['kernel_sizes'],
            conf['max_pools'] if conf.get('max_pools', False) else [1] * len(conf['n_channels']),
            conf['strides'] if conf.get('strides', False) else [1] * len(conf['n_channels']),
            conf['paddings'] if conf.get('paddings', False) else [0] * len(conf['n_channels']),
            conf['dropout'] if conf.get('dropout', False) else [0] * len(conf['n_channels']),
            conf['batch_norm'] if conf.get('batch_norm', False) else [None] * len(conf['n_channels'])
        )
        use_batch_norm = conf.get('use_batch_norm', False)
        pre_permute = conf.get('pre_permute', False)
        post_permute = conf.get('post_permute', False)
        self.unit_norm = conf.get('unit_norm', False)
        last_layer_activation = conf.get('last_layer_activation', True)

        if conf.get('conv_type', 'encode') == 'encode':
            conv = nn.Conv2d
        elif conf.get('conv_type', 'encode') == 'decode':
            conv = nn.ConvTranspose2d
        else:
            raise NotImplementedError(f'{conf["conv_type"]} conv type not implemented')

        if conf.get('activation_type', 'relu') == 'relu':
            activation = partial(nn.ReLU, inplace=True)
        elif conf.get('activation_type', 'relu') == 'leaky_relu':
            activation = partial(nn.LeakyReLU, 0.2, inplace=True)
        else:
            raise NotImplementedError(f'{conf["activation_type"]} activation type not implemented')

        if pre_permute:
            self.pre_permute = Permute(input_shape, *pre_permute)
            input_shape = self.pre_permute.output_shape
        else:
            self.pre_permute = None

        conv_layers = list()
        cur_channels = input_shape[0]
        for n_channels, kernel_size, max_pool, stride, padding, dropout, batch_norm in conv_params:
            conv_layers.append(conv(cur_channels, n_channels, kernel_size, stride, padding))
            if batch_norm:
                conv_layers.append(nn.BatchNorm2d(n_channels))
            conv_layers.append(activation())
            if dropout > 0:
                conv_layers.append(nn.Dropout(dropout))
            cur_channels = n_channels
            if max_pool == 'global':
                conv_layers.append(nn.AdaptiveMaxPool2d(1))
            elif max_pool > 1:
                conv_layers.append(nn.MaxPool2d(max_pool, max_pool))

        if not last_layer_activation:
            conv_layers.pop(-1)

        self.conv_net = nn.Sequential(*conv_layers)
        input_shape = get_output_shape(self.conv_net, input_shape)

        if post_permute:
            self.post_permute = Permute(input_shape, *post_permute)
            input_shape = self.post_permute.output_shape
        else:
            self.post_permute = None

        self.output_shape = input_shape
        self.output_size = reduce((lambda x, y: x * y), self.output_shape)

    def forward(self, x):
        if self.unit_norm:
            x = x.float() / 255.

        if self.pre_permute:
            x = self.pre_permute(x)

        x = self.conv_net(x)

        if self.post_permute:
            x = self.post_permute(x)
        return x


#LEGACY
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


#LEGACY
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


#LEGACY
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


#LEGACY
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
    def __init__(self, input_shape, n, *args):
        super().__init__()
        self.register_buffer('one_hot', torch.eye(n))
        self.output_shape = (*input_shape, n)

    def forward(self, x):
        return F.embedding(x, self.one_hot)


class DenseEmbed(nn.Embedding):
    def __init__(self, input_shape, n, dim, *args, **kwargs):
        super().__init__(n, dim, *args, **kwargs)
        self.output_shape = (*input_shape, dim)


class MultiEmbeddingNetwork(nn.Module):
    def __init__(self, input_shape, conf):
        super().__init__()
        assert isinstance(input_shape, dict), 'input_shape for MultiEmbeddingNetwork layer must be dict'

        embed_dims = conf['embed_dims']
        embed_n = conf['embed_n']
        self.embed_names = conf['embed_names']
        fc_layers = conf.get('fc_layers', [])

        embed_type = conf.get('embed_type', 'embed')
        if embed_type == 'embed':
            embed_l = DenseEmbed
        elif embed_type == 'ohe':
            embed_l = OneHot
        else:
            raise NotImplementedError(f'{embed_type} embeddings is not implemented')
        embed_layers = dict()
        for i, name in enumerate(self.embed_names):
            embed_layers[name] = embed_l(input_shape[name], embed_n[i], embed_dims[i])
        self.embed_layers = nn.ModuleDict(embed_layers)

        fc_in = sum(embed_dims)
        fc = list()
        for fc_out in fc_layers:
            fc.append(nn.Linear(fc_in, fc_out))
            fc.append(nn.ReLU())
            fc_in = fc_out
        self.fc_net = nn.Sequential(*fc)

        self.output_shape = (fc_in,)
        self.output_size = fc_in

    def forward(self, x):
        embeds = list()
        for n in self.embed_names:
            embeds.append(self.embed_layers[n](x[n].int()))
        embeds = torch.cat(embeds, dim=-1)
        return self.fc_net(embeds)


#LEGACY
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
        state_encoding = self.state_encoder.forward(states_and_goals['image'])
        return torch.cat([state_encoding, goal_encoding], dim=1)


#LEGACY
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


#LEGACY
class RemoveOuterWalls(nn.Module):
    def __init__(self, model, tile_size):
        super().__init__()
        self.tile_size = tile_size
        self.model = model
        if hasattr(model, 'output_size'):
            self.output_size = model.output_size

    def forward(self, x):
        return self.model(x[:, self.tile_size: -self.tile_size, self.tile_size: -self.tile_size])


#LEGACY
class NormEncoder(nn.Module):
    def __init__(self, model, eps=1e-9):
        super().__init__()
        self.eps = eps
        self.model = model
        self.output_size = model.output_size

    def forward(self, out: torch.Tensor):
        out = self.model(out)
        return out / (out.pow(2).sum(dim=1) + self.eps).pow(0.5).unsqueeze(-1).expand(*out.size())


#LEGACY
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


#LEGACY
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


class Sequence(nn.Module):
    def __init__(self, input_shape, conf):
        super().__init__()
        self.models_names = conf['transformations']
        models = dict()

        for model_name in self.models_names:
            model = get_encoder(None, conf[model_name], input_shape)
            input_shape = model.output_shape
            models[model_name] = model

        self.models = nn.ModuleDict(models)
        self.output_shape = input_shape

    def forward(self, x):
        for model_name in self.models_names:
            x = self.models[model_name](x)
        return x


class Parallel(nn.Module):
    def __init__(self, input_shape, conf):
        super().__init__()
        self.models_names = conf['transformations']
        models = dict()
        out_shapes = dict()

        for model_name in self.models_names:
            model = get_encoder(None, conf[model_name], input_shape)
            models[model_name] = model
            out_shapes[model_name] = model.output_shape

        self.models = nn.ModuleDict(models)
        self.output_shape = out_shapes

    def forward(self, x):
        outputs = dict()
        for model_name in self.models_names:
            outputs[model_name] = self.models[model_name](x[model_name])
        return outputs


#LEGACY
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


def get_encoder(grid_size, config, input_shape=None):
    if config['state_encoder_type'] == 'simple_mlp':
        state_size = 3 * (grid_size - 2) ** 2
        state_encoder = Flattener(MLP(state_size, config['hidden_layers_sizes']))
    elif config['state_encoder_type'] == 'mlp_encoder':
        state_encoder = MLPEncoder(input_shape, config)
    elif config['state_encoder_type'] == 'simple_cnn':
        state_encoder = SimpleCNN(grid_size, config)
    elif config['state_encoder_type'] == 'cnn_encoder':
        state_encoder = SimpleCNNEncoder(input_shape, config)
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
    elif config['state_encoder_type'] == 'sequence_rnn':
        sequence_encoder = Sequence(input_shape, config['inner_encoder'])
        state_encoder = RNNEncoder(sequence_encoder,
                                   config.get('rnn_output', sequence_encoder.output_size),
                                   config.get('rnn_num_layers', 1))
    elif config['state_encoder_type'] == 'multi_embeddings':
        state_encoder = MultiEmbeddingNetwork(input_shape, config)
    elif config['state_encoder_type'] == 'cnn_hindsight_encoder':
        goal_encoder = get_encoder(grid_size, config['goal_encoder'], input_shape)
        first_cnn_encoder = get_encoder(grid_size, config['first_cnn_encoder'], input_shape)
        config['second_cnn_encoder']['input_channels'] = goal_encoder.output_size + first_cnn_encoder.output_shape[0]
        grid_size = first_cnn_encoder.output_shape[1]
        second_cnn_encoder = get_encoder(grid_size, config['second_cnn_encoder'], input_shape)
        grid_size = second_cnn_encoder.output_size
        config['fc_encoder']['input_size'] = second_cnn_encoder.output_size
        fc_encoder = get_encoder(grid_size, config['fc_encoder'], input_shape)
        state_encoder = ConvGoalStateEncoder(goal_encoder, first_cnn_encoder, second_cnn_encoder, fc_encoder)
    elif config['state_encoder_type'] == 'sequence':
        state_encoder = Sequence(input_shape, config)
    elif config['state_encoder_type'] == 'parallel':
        state_encoder = Parallel(input_shape, config)
    elif config['state_encoder_type'] == 'permute':
        state_encoder = Permute(input_shape, *config['channels'])
    elif config['state_encoder_type'] == 'split':
        state_encoder = Split(input_shape, config)
    elif config['state_encoder_type'] == 'concat':
        state_encoder = Concat(input_shape, config)
    elif config['state_encoder_type'] == 'sigmoid':
        state_encoder = Sigmoid(input_shape, config)
    elif config['state_encoder_type'] == 'tanh':
        state_encoder = Tanh(input_shape, config)
    elif config['state_encoder_type'] == 'flatten':
        state_encoder = Flatten(input_shape, config)
    elif config['state_encoder_type'] == 'reshape':
        state_encoder = Reshape(input_shape, config)
    elif config['state_encoder_type'] == 'dict_choose':
        state_encoder = DictChoose(input_shape, config)
    elif config['state_encoder_type'] == 'noise_vec':
        state_encoder = NoiseVec(input_shape, config)
    else:
        raise AttributeError(f"unknown nn_type '{config['master.state_encoder_type']}'")

    if config.get('normalize', False):
        state_encoder = NormEncoder(state_encoder)

    return state_encoder
