"""model.py"""
import math

import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.nn.init as init


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = std.data.new(std.size()).normal_()
    return mu + std*eps


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, input_length, z_dim=10, kernel_size = 13, n_filters=128, stride=2):
        super(BetaVAE_H, self).__init__()
        self.z_dim = z_dim
        self.kernel_size = kernel_size
        self.input_length = input_length
        self.n_filters = n_filters
        self.stride = stride
        encoder_padding = int((kernel_size - 1) / 2)  # to exactly reduce the sequence length by half after conv
        self.conv1d1_length = math.floor(((input_length + 2 * encoder_padding - (kernel_size-1) - 1) / stride) + 1)
        self.conv1d2_length = math.floor(((self.conv1d1_length + 2 * encoder_padding - (kernel_size-1) - 1) / stride) + 1)
        self.conv1d3_length = math.floor(((self.conv1d2_length + 2 * encoder_padding - (kernel_size-1) - 1) / stride) + 1)
        # self.conv1d4_length = math.floor(((self.conv1d3_length - (kernel_size-1)) / stride) + 1)
        # self.conv1d5_length = math.floor(((self.conv1d4_length - (kernel_size-1)) / stride) + 1)

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=n_filters, kernel_size=(self.kernel_size, ), stride=(2,), padding=encoder_padding),  # Enhancer: B, 128, 1500
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=(self.kernel_size,), stride=(2,), padding=encoder_padding),  # Enhancer: B, 128, 750
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=(self.kernel_size,), stride=(2,), padding=encoder_padding),  # Enhancer: B, 128, 375
            nn.LeakyReLU(),
            # nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=(self.kernel_size,), stride=(2,)),
            # nn.LeakyReLU(),
            # nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=(self.kernel_size,), stride=(2,)),
            # nn.LeakyReLU(),
            View((-1, self.conv1d3_length * n_filters)),  # B, 48000
            nn.Linear(self.conv1d3_length * n_filters, z_dim*2),  # B, z_dim * 2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, self.conv1d3_length * n_filters),  # B, 48000
            View((-1, n_filters, self.conv1d3_length)),  # B, 128, 375
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=n_filters, out_channels=n_filters, kernel_size=(self.kernel_size,), stride=(2,), padding=encoder_padding, output_padding=1),  # Enhancer: B, 128, 750
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=n_filters, out_channels=n_filters, kernel_size=(self.kernel_size,), stride=(2,), padding=encoder_padding, output_padding=1),  # Enhancer: B, 128, 1500
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=n_filters, out_channels=4, kernel_size=(self.kernel_size,), stride=(2,), padding=encoder_padding, output_padding=1),  # Enhancer: B, 4, 3000

            # nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            # nn.ReLU(True),
            # nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            # nn.ReLU(True),
            # nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            # nn.ReLU(True),
            # nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            # nn.ReLU(True),
            # nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class BetaVAE_B(BetaVAE_H):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, z_dim=10, nc=1):
        super(BetaVAE_B, self).__init__()
        self.nc = nc
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  8,  8
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  4,  4
            nn.ReLU(True),
            View((-1, 32*4*4)),                  # B, 512
            nn.Linear(32*4*4, 256),              # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 32*4*4),              # B, 512
            nn.ReLU(True),
            View((-1, 32, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1), # B,  nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z).view(x.size())

        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


if __name__ == '__main__':
    pass