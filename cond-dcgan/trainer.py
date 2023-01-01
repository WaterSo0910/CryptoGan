import torch
import logging
from utils import plot_res, plot_dis
from collections import defaultdict
import torch.utils.data as td
from torchmetrics import MeanAbsolutePercentageError
import os

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        train_dataloader: td.DataLoader,
        netD: torch.nn.Module,
        netG: torch.nn.Module,
        optimizerD: torch.optim.Optimizer,
        optimizerG: torch.optim.Optimizer,
        nc: int,
        nz: int,
        timeseries_size: int,
        num_epochs: int,
        device: torch.device,
        criterion: torch.nn.Module,
        real_label=0.9,
        fake_label=0.1,
    ) -> None:
        self.train_dataloader = train_dataloader
        self.netD = netD
        self.netG = netG
        self.nc = nc
        self.nz = nz
        self.optimizerD = optimizerD
        self.optimizerG = optimizerG
        self.timeseries_size = timeseries_size
        self.num_epochs = num_epochs
        self.device = device
        self.criterion = criterion
        self.real_label = 0.9
        self.fake_label = 0.1
        self.mape = MeanAbsolutePercentageError()

    def d_step(self, args, info: torch.Tensor, real: torch.Tensor, noise: torch.Tensor):
        self.netD.zero_grad()
        batch_size = real.size(0)
        label = torch.full(
            (batch_size,), self.real_label, dtype=torch.float, device=self.device
        )
        real = 0.9 * real + 0.1 * torch.randn((real.size()), device=self.device)
        output = self.netD(real, info).view(-1)
        errD_real = self.criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()
        ## D: Train with all-fake(G) batch
        info_g = info.view(-1, self.timeseries_size, self.nc)

        fake = self.netG(noise, info_g)
        label.fill_(self.fake_label)
        fake = 0.9 * fake + 0.1 * torch.randn((fake.size()), device=self.device)
        output = self.netD(fake.detach(), info.detach()).view(-1)
        errD_fake = self.criterion(output, label)
        errD = errD_real + errD_fake
        errD_fake.backward()
        self.optimizerD.step()

        return errD

    def g_step(self, args, info: torch.Tensor, real: torch.Tensor, noise: torch.Tensor):
        batch_size = real.size(0)
        info_g = info.view(-1, self.timeseries_size, self.nc)
        fake = self.netG(noise, info_g)
        mape = self.mape(fake.detach().cpu(), real.detach().cpu())
        label = torch.full(
            (batch_size,), self.real_label, dtype=torch.float, device=self.device
        )
        self.netG.zero_grad()
        label.fill_(self.real_label)  # fake labels are real for generator cost
        output = self.netD(fake, info).view(-1)
        errG = self.criterion(output, label)
        errG.backward()
        self.optimizerG.step()
        return errG, mape

    # Training Loop
    def train(
        self,
        args,
    ):
        checkpoint = {
            "args": args.__dict__,
            "G_losses": defaultdict(list),
            "D_losses": defaultdict(list),
            "sample_ts": [],
            "counters": {
                "t": None,
                "epoch": None,
            },
            "g_state": None,
            "d_state": None,
            "g_optim_state": None,
            "d_optim_state": None,
        }
        t = 0
        D_losses, G_losses = [], []
        mapes = []
        # For each epoch
        for epoch in range(self.num_epochs):
            logger.info("Starting epoch {}".format(epoch))
            for i, data in enumerate(self.train_dataloader, 0):
                info = data[0].to(self.device).view(-1, self.nc, self.timeseries_size)
                mask = data[1]
                real = data[2].to(self.device).view(-1, self.nc, self.timeseries_size)
                batch_size = real.size(0)
                noise = torch.randn(batch_size, self.nz, device=self.device).view(
                    -1, self.nz, 1
                )
                ## D: Train with all-real batch
                d_loss = self.d_step(args, info, real, noise)
                g_loss, mape = self.g_step(args, info, real, noise)
                D_losses.append(d_loss)
                G_losses.append(g_loss)
                mapes.append(mape)
                if (t % args.print_every == 0) or (
                    (epoch == self.num_epochs - 1)
                    and (i == len(self.train_dataloader) - 1)
                ):
                    with torch.no_grad():
                        fake = (
                            self.netG(noise, info.view(-1, self.timeseries_size, 1))
                            .detach()
                            .cpu()
                        )
                        fake, real = (
                            fake[0].view(-1).tolist(),
                            real[0].view(-1).tolist(),
                        )
                        mask = mask[0].view(-1).tolist()
                    params = {"iters": t, "epoch": epoch}
                    plot_res(args, fake, real, mask, **params)
                    plot_dis(args, fake, real, **params)
                    logger.info("t = {} / {}".format(t + 1, args.num_iterations))
                    logger.info("  [D] Loss: {:.3f}".format(d_loss))
                    logger.info("  [G] Loss: {:.3f}".format(g_loss))
                    logger.info("  [G] MAPE: {:.3f}".format(mape))
                    logger.info(
                        "  [D] Total Loss: {:.3f}".format(sum(D_losses) / len(D_losses))
                    )
                    logger.info(
                        "  [G] Total Loss: {:.3f}".format(sum(G_losses) / len(G_losses))
                    )
                    logger.info(
                        "  [G] Total MAPE: {:.3f}".format(sum(mapes) / len(mapes))
                    )
                if t > 0 and t % args.checkpoint_every == 0:
                    checkpoint["counters"]["t"] = t
                    checkpoint["counters"]["epoch"] = epoch
                    checkpoint["sample_ts"].append(t)
                    checkpoint["g_state"] = self.netG.state_dict()
                    checkpoint["g_optim_state"] = self.optimizerG.state_dict()
                    checkpoint["d_state"] = self.netD.state_dict()
                    checkpoint["d_optim_state"] = self.optimizerD.state_dict()
                    checkpoint_path = os.path.join(
                        args.checkpoint_path, f"checkpoint_with_model_{t}_.pt"
                    )
                    logger.info("t = {} / {}".format(t + 1, args.num_iterations))
                    logger.info(
                        "  [D] Total Loss: {:.3f}".format(sum(D_losses) / len(D_losses))
                    )
                    logger.info(
                        "  [G] Total Loss: {:.3f}".format(sum(G_losses) / len(G_losses))
                    )
                    logger.info(
                        "  [G] Total MAPE: {:.3f}".format(sum(mapes) / len(mapes))
                    )
                    logger.info("Saving checkpoint to {}".format(checkpoint_path))
                    torch.save(checkpoint, checkpoint_path)
                    logger.info("Done.")
                t += 1
