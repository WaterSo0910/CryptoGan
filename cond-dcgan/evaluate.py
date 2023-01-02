from models import Generator
from attrdict import AttrDict
from torchmetrics import MeanAbsolutePercentageError
import torch
import os
from data import data_loader

from tqdm import tqdm


def get_generator(checkpoint):
    args = AttrDict(checkpoint["args"])
    generator = Generator(
        ngpu=1,
        input_dim=args.input_dim,
        noise_dim=args.noise_dim,
        ngf=args.ngf,
        seq_len=args.seq_len,
    )
    generator.load_state_dict(checkpoint["g_state"])
    generator.cuda()
    generator.train()
    return generator


def filling(
    info: torch.Tensor,
    mask: torch.Tensor,
    method="random",
) -> torch.Tensor:
    if method == "random":
        return info
    elif method == "bfill":
        bfill_val = info.roll(dims=(1), shifts=(1))
        info[~mask.bool()] = bfill_val[~mask.bool()]
        return info


def evaluate(args, loader, generator, num_samples):
    mape = MeanAbsolutePercentageError()
    MAPE = {
        "rand": [],
        "fake": [],
        "bfill": [],
    }
    print("Start evaluation")
    with tqdm(total=len(loader)) as pbar:
        for batch in loader:
            batch = [
                tensor.cuda().view(-1, args.input_dim, args.seq_len) for tensor in batch
            ]
            (
                info,
                mask,
                real,
            ) = batch
            batch_size = real.size(0)
            for _ in range(num_samples):
                info_g = info.view(-1, args.seq_len, args.input_dim)
                noise = (
                    torch.randn(batch_size, args.noise_dim)
                    .cuda()
                    .view(-1, args.noise_dim, 1)
                )
                fake = generator(noise, info_g)
                fake_input = fake.clone()
                fake_input[mask.bool()] = real[mask.bool()]
                rand_input = filling(info, mask, method="random")
                bfill_input = filling(info, mask, method="bfill")
                MAPE["fake"].append(
                    mape(fake_input.detach().cpu(), real.detach().cpu())
                )
                MAPE["rand"].append(
                    mape(rand_input.detach().cpu(), real.detach().cpu())
                )
                MAPE["bfill"].append(
                    mape(bfill_input.detach().cpu(), real.detach().cpu())
                )
            pbar.update(1)
    return MAPE


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="checkpoints", type=str)
parser.add_argument("--num_samples", default=20, type=int)


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [os.path.join(args.model_path, file_) for file_ in filenames]
    else:
        paths = [args.model_path]
    for path in paths:
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint["args"])
        dset, loader = data_loader(_args, mode="test")
        mapes = evaluate(_args, loader, generator, args.num_samples)
        print(
            "Seq Len: {}, Mask rate: {} (path={})".format(
                _args.seq_len, _args.mask_rate, path
            )
        )
        for k, v in mapes.items():
            print("\tMape({}): {:.4f}".format(k, sum(v) / len(v)))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
