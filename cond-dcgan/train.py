from torchsummary import summary

import argparse
import os
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as td
import numpy as np
import torch.nn.functional as F
import plotly.express as px
import os
import logging
import sys

import numpy as np
import pandas as pd

# My pkg
from data import CrytoDataset
from models import Generator, Discriminator
from trainer import Trainer
from utils import plot_dist

manualSeed = 880910
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


# Dataset options
parser.add_argument("--datapath", default="./data/Binance_ETHUSDT_1h.csv", type=str)
parser.add_argument("--image_path", default="images", type=str)
parser.add_argument("--dist_path", default="dist", type=str)
parser.add_argument("--mask_at", default="random", type=str)  # random / end
parser.add_argument("--mask_rate", default=0.4, type=float)
parser.add_argument("--test_size", default=0.3, type=float)
parser.add_argument("--seq_len", default=64, type=int)
parser.add_argument("--skip", default=1, type=int)

# Optimization
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--num_iterations", default=10000, type=int)
parser.add_argument("--num_epochs", default=200, type=int)

# Model Options
parser.add_argument("--input_dim", default=1, type=int)
parser.add_argument("--noise_dim", default=10, type=int)
parser.add_argument("--ngf", default=64, type=int)
parser.add_argument("--ndf", default=64, type=int)
parser.add_argument("--learning_rate", default=2e-4, type=float)

# Print option
parser.add_argument("--print_every", default=1000, type=int)

ngpu = 1


def main(args):
    os.makedirs(args.image_path, exist_ok=True)
    os.makedirs(args.dist_path, exist_ok=True)
    # Create the dataset
    train_dataset = CrytoDataset(
        path=args.datapath,
        mode="train",
        test_size=args.test_size,
        window_size=args.seq_len,
        mask_prob=args.mask_rate,
    )
    test_dataset = CrytoDataset(
        path=args.datapath,
        mode="test",
        test_size=args.test_size,
        window_size=args.seq_len,
        mask_prob=args.mask_rate,
    )
    train_dataloader = td.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_dataloader = td.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True
    )
    device = torch.device("cuda:0")

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    # Create the generator
    netG = Generator(
        ngpu=ngpu,
        input_dim=args.input_dim,
        noise_dim=args.noise_dim,
        ngf=args.ngf,
        seq_len=args.seq_len,
    ).to(device)
    netG.apply(weights_init)
    logger.info(netG)

    # Create the Discriminator
    netD = Discriminator(
        ngpu=ngpu,
        input_dim=args.input_dim,
        ndf=args.ndf,
    ).to(device)
    netD.apply(weights_init)
    logger.info(netD)

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(args.batch_size, args.noise_dim, 1, 1, device=device)

    optimizerD = optim.Adam(
        netD.parameters(), lr=args.learning_rate, betas=(0.5, 0.999)
    )
    optimizerG = optim.Adam(
        netG.parameters(), lr=args.learning_rate, betas=(0.5, 0.999)
    )
    trainer = Trainer(
        train_dataloader=train_dataloader,
        netD=netD,
        netG=netG,
        optimizerD=optimizerD,
        optimizerG=optimizerG,
        device=device,
        criterion=criterion,
        nc=args.input_dim,
        nz=args.noise_dim,
        timeseries_size=args.seq_len,
        num_epochs=args.num_epochs,
    )

    G_losses, D_losses = trainer.train(
        args=args,
        real_label=0.9,
        fake_label=0.1,
        print_every=args.print_every,
    )
    logger.info("Plot distribution")
    plot_dist(G_losses, D_losses)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
