import torch
import logging
from utils import plot_res, plot_dis
import torch.utils.data as td

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

    # Training Loop
    def train(
        self,
        args,
        print_every=100,
        real_label=0.9,
        fake_label=0.1,
    ):
        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0
        logger.info("Start training ...")
        # For each epoch
        for epoch in range(self.num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(self.train_dataloader, 0):
                ## D: Train with all-real batch
                self.netD.zero_grad()
                info = data[0].to(self.device).view(-1, self.nc, self.timeseries_size)
                mask = data[1]
                real_cpu = (
                    data[2].to(self.device).view(-1, self.nc, self.timeseries_size)
                )
                b_size = real_cpu.size(0)
                label = torch.full(
                    (b_size,), real_label, dtype=torch.float, device=self.device
                )
                real_cpu = 0.9 * real_cpu + 0.1 * torch.randn(
                    (real_cpu.size()), device=self.device
                )
                output = self.netD(real_cpu, info).view(-1)
                errD_real = self.criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                ## D: Train with all-fake(G) batch
                noise = torch.randn(b_size, self.nz, device=self.device).view(
                    -1, self.nz, 1
                )
                info_g = data[0].to(self.device).view(-1, self.timeseries_size, self.nc)
                fake = self.netG(noise, info_g)
                label.fill_(fake_label)
                fake = 0.9 * fake + 0.1 * torch.randn((fake.size()), device=self.device)
                output = self.netD(fake.detach(), info.detach()).view(-1)
                errD_fake = self.criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                self.optimizerD.step()

                self.netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                output = self.netD(fake, info).view(-1)
                errG = self.criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                self.optimizerG.step()

                if (iters % print_every == 0) or (
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
                            real_cpu[0].view(-1).tolist(),
                        )
                        mask = mask[0].view(-1).tolist()
                    params = {"iters": iters, "epoch": epoch}
                    plot_res(args, fake, real, mask, **params)
                    plot_dis(args, fake, real, **params)
                    logger.info(
                        "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                        % (
                            epoch,
                            self.num_epochs,
                            i,
                            len(self.train_dataloader),
                            errD.item(),
                            errG.item(),
                            D_x,
                            D_G_z1,
                            D_G_z2,
                        )
                    )
                iters += 1
                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())
        return G_losses, D_losses
