import torch.nn as nn
import torch

# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu, input_dim, noise_dim, ngf, seq_len):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.noise_layers = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose1d(noise_dim, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
        )
        self.info_layers = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose1d(seq_len, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
        )
        self.main = nn.Sequential(
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose1d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose1d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose1d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose1d(ngf, input_dim, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input, info):
        x1 = self.noise_layers(input)
        x2 = self.info_layers(info)
        x = torch.cat([x1, x2], 1)
        return self.main(x)


# Discriminator Code
class Discriminator(nn.Module):
    def __init__(self, ngpu, input_dim, ndf):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.input_layers = nn.Sequential(
            nn.Conv1d(input_dim, int(ndf / 2), 4, 2, 1, bias=False),
            nn.BatchNorm1d(int(ndf / 2)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.info_layers = nn.Sequential(
            nn.Conv1d(input_dim, int(ndf / 2), 4, 2, 1, bias=False),
            nn.BatchNorm1d(int(ndf / 2)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.main = nn.Sequential(
            nn.Conv1d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32
            nn.Conv1d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16
            nn.Conv1d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4
            nn.Conv1d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input, info):
        x1 = self.input_layers(input)
        x2 = self.info_layers(info)
        x = torch.cat([x1, x2], 1)
        return self.main(x)
