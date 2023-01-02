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
from data import data_loader
from models import Generator, Discriminator, LSTM
from trainer import Trainer, LSTMTrainer
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
parser.add_argument("--datapath", default="../data/Binance_ETHUSDT_1h.csv", type=str)
parser.add_argument("--image_path", default="images", type=str)
parser.add_argument("--dist_path", default="dist", type=str)
parser.add_argument("--mask_where", default="random", type=str)  # random / end
parser.add_argument("--mask_rate", default=0.4, type=float)
parser.add_argument("--test_size", default=0.3, type=float)
parser.add_argument("--seq_len", default=64, type=int)
parser.add_argument("--skip", default=1, type=int)

# Optimization
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--num_iterations", default=10000, type=int)
parser.add_argument("--num_epochs", default=200, type=int)

# Model Options
parser.add_argument("--model", default="cond-dcgan", type=str)
parser.add_argument("--input_dim", default=1, type=int)
parser.add_argument("--noise_dim", default=10, type=int)
parser.add_argument("--ngf", default=64, type=int)
parser.add_argument("--ndf", default=64, type=int)
parser.add_argument("--learning_rate", default=2e-4, type=float)

# LSTM option
parser.add_argument("--num_layers", default=1, type=int)
parser.add_argument("--hidden_size", default=64, type=int)
parser.add_argument("--dropout", default=0.0, type=float)


# Output option
parser.add_argument("--print_every", default=200, type=int)
parser.add_argument("--checkpoint_every", default=1000, type=int)
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--checkpoint_path", default="checkpoints", type=str)

ngpu = 1


def main(args):
    os.makedirs(args.image_path, exist_ok=True)
    os.makedirs(args.dist_path, exist_ok=True)
    os.makedirs(args.checkpoint_path, exist_ok=True)
    # Create the dataset
    _, train_dataloader = data_loader(args, mode="train")
    _, test_dataloader = data_loader(args, mode="test")

    device = torch.device("cuda:0")
    if args.model == "cond-dcgan":

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find("Conv") != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find("BatchNorm") != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
            elif classname.find("Linear") != -1:
                nn.init.kaiming_normal_(m.weight)

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
        optimizerD = optim.Adam(
            netD.parameters(), lr=args.learning_rate, betas=(0.5, 0.999)
        )
        optimizerG = optim.Adam(
            netG.parameters(), lr=args.learning_rate, betas=(0.5, 0.999)
        )
        if args.checkpoint is not None:
            restore_path = args.checkpoint
            logger.info("Restoring from checkpoint {}".format(restore_path))
            checkpoint = torch.load(restore_path)
            netG.load_state_dict(checkpoint["g_state"])
            netD.load_state_dict(checkpoint["d_state"])
            optimizerG.load_state_dict(checkpoint["g_optim_state"])
            optimizerD.load_state_dict(checkpoint["d_optim_state"])

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
            real_label=0.9,
            fake_label=0.1,
        )

        G_losses, D_losses = trainer.train(
            args=args,
        )
        logger.info("Plot distribution")
        plot_dist(G_losses, D_losses)
    elif args.model == "lstm":
        obs_len = int(args.seq_len * (1 - args.mask_rate))
        pred_len = args.seq_len - obs_len
        model = LSTM(
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
        optimizer = optim.Adam(
            model.parameters(), lr=args.learning_rate, betas=(0.5, 0.999)
        )
        criterion = nn.MSELoss()
        if args.checkpoint is not None:
            restore_path = args.checkpoint
            logger.info("Restoring from checkpoint {}".format(restore_path))
            checkpoint = torch.load(restore_path)
            model.load_state_dict(checkpoint["state"])
            optimizer.load_state_dict(checkpoint["optim_state"])
        trainer = LSTMTrainer(
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            model=model,
            optimizer=optimizer,
            pred_len=pred_len,
            obs_len=obs_len,
            num_epochs=args.num_epochs,
            device=device,
            criterion=criterion,
        )
        trainer.train(args)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
