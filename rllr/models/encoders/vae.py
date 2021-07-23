from functools import reduce

import torch
from torch import nn

from rllr.models.encoders import Permute


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
