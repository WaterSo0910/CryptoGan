import torch
import logging
from utils import plot_res, plot_dis
from collections import defaultdict
import torch.utils.data as td
from torchmetrics import MeanAbsolutePercentageError
import os
from typing import Tuple
from utils import plot_dist

logger = logging.getLogger(__name__)


class LSTMTrainer:
    def __init__(
        self,
        train_dataloader: td.DataLoader,
        test_dataloader: td.DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        obs_len: int,
        pred_len: int,
        num_epochs: int,
        device: torch.device,
        criterion: torch.nn.Module,
    ) -> None:
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.mape = MeanAbsolutePercentageError().to(device)

    def train(self, args):
        mapes, losses = [], []
        best_loss = None
        t = 0
        checkpoint = {
            "args": args.__dict__,
            "losses": [],
            "sample_ts": [],
            "counters": {
                "t": None,
                "epoch": None,
            },
            "state": None,
            "optim_state": None,
        }
        for epoch in range(self.num_epochs):
            logger.info("Starting epoch {}".format(epoch))
            for i, batch in enumerate(self.train_dataloader, 0):
                batch = [tensor.cuda() for tensor in batch]
                (info, mask, real) = batch
                batch_size = real.size(0)
                assert info.size() == torch.Size(
                    [batch_size, self.obs_len + self.pred_len]
                )
                assert mask.size() == torch.Size(
                    [batch_size, self.obs_len + self.pred_len]
                )
                assert real.size() == torch.Size(
                    [batch_size, self.obs_len + self.pred_len]
                )
                self.model.train()
                input = real[:, : self.obs_len]
                outs = torch.zeros(batch_size, self.pred_len).to(self.device)
                for p in range(self.pred_len):
                    out = self.model(input[:, :, None])
                    outs[:, p] = out.view(-1)
                    input = torch.concat((input, out), dim=1)
                    input = input[:, 1:]
                out_gt = real[:, self.obs_len :]
                assert out_gt.size() == torch.Size(
                    [batch_size, self.pred_len]
                ), out_gt.size()
                self.optimizer.zero_grad()
                assert out_gt.size() == outs.size(), "{}{}".format(
                    out_gt.size(), outs.size()
                )
                loss = self.criterion(outs, out_gt)
                mape = self.mape(outs, out_gt)
                loss.backward()
                self.optimizer.step()
                losses.append(loss)
                mapes.append(mape)
                t += 1
            if epoch % args.print_every == 0:
                logger.info("t = {} / {}".format(t + 1, args.num_iterations))
                logger.info("  Loss: {:.3f}".format(loss))
                logger.info("  MAPE: {:.3f}".format(mape))
                logger.info("  Total Loss: {:.3f}".format(sum(losses) / len(losses)))
                logger.info("  Total MAPE: {:.3f}".format(sum(mapes) / len(mapes)))

            val_loss, val_mape = self.validate(args, epoch=epoch + 1, iters=t + 1)
            if best_loss == None or val_mape < best_loss:
                best_loss = val_mape
                checkpoint["counters"]["t"] = t
                checkpoint["counters"]["epoch"] = epoch
                checkpoint["sample_ts"].append(t)
                checkpoint["losses"].append(val_loss)
                checkpoint["state"] = self.model.state_dict()
                checkpoint["optim_state"] = self.optimizer.state_dict()
                checkpoint_path = os.path.join(
                    args.checkpoint_path, f"lstm_model_{t}.pt"
                )
                logger.info(
                    "Best model update ... t = {} / {}".format(
                        t + 1, args.num_iterations
                    )
                )
                logger.info("\tAvg Loss: {:.3f}".format(val_loss))
                logger.info("\tAvg MAPE: {:.3f}".format(val_mape))
                logger.info("Saving checkpoint to {}".format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                logger.info("Done.")

    def validate(self, args, epoch, iters) -> Tuple[float, float]:
        losses = []
        mapes = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.test_dataloader, 0):
                batch = [tensor.cuda() for tensor in batch]
                (info, mask, real) = batch
                batch_size = real.size(0)
                assert info.size() == torch.Size(
                    [batch_size, self.obs_len + self.pred_len]
                )
                assert mask.size() == torch.Size(
                    [batch_size, self.obs_len + self.pred_len]
                )
                assert real.size() == torch.Size(
                    [batch_size, self.obs_len + self.pred_len]
                )
                input = real[:, : self.obs_len]
                outs = torch.zeros(batch_size, self.pred_len).to(self.device)
                for p in range(self.pred_len):
                    out = self.model(input[:, :, None])
                    outs[:, p] = out.view(-1)
                    input = torch.concat((input, out), dim=1)
                    input = input[:, 1:]
                out_gt = real[:, self.obs_len :]
                assert out_gt.size() == torch.Size([batch_size, self.pred_len])
                assert outs.size() == torch.Size([batch_size, self.pred_len])
                loss = self.criterion(outs, out_gt)
                mape = self.mape(outs, out_gt)
                losses.append(loss)
                mapes.append(mape)
                if i == 0:
                    plot_res(
                        args,
                        real[0, : self.obs_len].view(-1).tolist()
                        + outs[0, :].view(-1).tolist(),
                        real[0].view(-1).tolist(),
                        mask[0].view(-1).tolist(),
                        epoch=epoch,
                        iters=iters,
                    )
        return sum(losses) / len(losses), sum(mapes) / len(mapes)


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
        self.real_label = real_label
        self.fake_label = fake_label
        self.mape = MeanAbsolutePercentageError()
        self.l2_loss = torch.nn.MSELoss()

    def d_step(self, args, info: torch.Tensor, real: torch.Tensor, noise: torch.Tensor):
        self.optimizerD.zero_grad()
        batch_size = real.size(0)
        label = torch.full(
            (batch_size,), self.real_label, dtype=torch.float, device=self.device
        )
        real = 0.9 * real + 0.1 * torch.randn((real.size()), device=self.device)
        output = self.netD(real, info).view(-1)
        errD_real = self.criterion(output, label)
        errD_real.backward()
        ## D: Train with all-fake(G) batch
        info_g = info.view(-1, self.timeseries_size, self.nc)

        fake = self.netG(noise, info_g)
        label.fill_(self.fake_label)
        fake = 0.9 * fake + 0.1 * torch.randn((fake.size()), device=self.device)
        output = self.netD(fake.detach(), info.detach()).view(-1)
        errD_fake = self.criterion(output, label)
        errD_fake.backward()
        output = self.netD(fake.detach(), info.detach()).view(-1)
        errD = errD_real + errD_fake
        if args.l2_loss_weight > 0:
            l2_loss = self.l2_loss(output, label)
            l2_loss.backward()
            errD += l2_loss
        self.optimizerD.step()
        return errD

    def g_step(
        self,
        args,
        info: torch.Tensor,
        real: torch.Tensor,
        mask: torch.Tensor,
        noise: torch.Tensor,
    ):
        batch_size = real.size(0)
        info_g = info.view(-1, self.timeseries_size, self.nc)
        fake = self.netG(noise, info_g)
        label = torch.full(
            (batch_size,), self.real_label, dtype=torch.float, device=self.device
        )
        self.netG.zero_grad()
        label.fill_(self.real_label)  # fake labels are real for generator cost
        output = self.netD(fake, info).view(-1)
        errG = self.criterion(output, label)
        errG.backward()
        self.optimizerG.step()
        fake_input = fake.clone()
        fake_input[mask.bool()] = real[mask.bool()]
        mape = self.mape(fake_input.detach().cpu(), real.detach().cpu())
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
                mask = data[1].to(self.device).view(-1, self.nc, self.timeseries_size)
                real = data[2].to(self.device).view(-1, self.nc, self.timeseries_size)
                batch_size = real.size(0)
                noise = torch.randn(batch_size, self.nz, device=self.device).view(
                    -1, self.nz, 1
                )
                ## D: Train with all-real batch
                d_loss = self.d_step(args, info, real, noise)
                g_loss, mape = self.g_step(args, info, real, mask, noise)
                D_losses.append(d_loss.item())
                G_losses.append(g_loss.item())
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
                        args.checkpoint_path,
                        f"model_{t}_seq={args.seq_len}_mask={args.mask_rate}.pt",
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
        logger.info("Plot distribution")
        plot_dist(G_losses, D_losses, epoch=epoch, iters=t)
        return G_losses, D_losses
