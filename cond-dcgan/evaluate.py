from models import Generator, LSTM
from attrdict import AttrDict
from torchmetrics import MeanAbsolutePercentageError, MeanAbsoluteError
import torch
import os
from data import data_loader
from utils import plot_evaluation, plot_2_seqs
from tqdm import tqdm
import numpy as np


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(880910)


def get_lstm_1(checkpoint):
    args = AttrDict(checkpoint["args"])
    model = LSTM(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model.load_state_dict(checkpoint["state"])
    model.cuda()
    return model


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
    return generator


def filling(
    info: torch.Tensor,
    mask: torch.Tensor,
    method,
) -> torch.Tensor:
    """filling missing value

    Args:
        info (torch.Tensor): (batch, seq, 1)
        mask (torch.Tensor): (batch, seq, 1)
        method (str): bfill or mean
    Returns:
        torch.Tensor:  (batch, seq, 1)
    """
    if method == "mean":
        return info
    elif method == "bfill":
        bfill_val = torch.zeros_like(info)
        zeros = torch.zeros_like(info)
        bfill_val[mask.bool()] = info[mask.bool()]
        use_col = info[:, 0, :]

        for i in range(1, bfill_val.size(1)):
            if (~mask[:, i, 0].bool()).all():
                bfill_val[:, i, :] = use_col
            else:
                use_col = bfill_val[:, i, :]
        return bfill_val
    raise ValueError("Filling method should be mean or bfill, but {}.".format(method))


def evaluate(args, loader, generator, num_samples, path):
    mae = MeanAbsoluteError()
    MAE = {
        "mean": [],
        "bfill": [],
        "fake": [],
    }
    print("Start evaluation")
    with tqdm(total=len(loader)) as pbar:
        t = 0
        for batch in loader:
            batch = [
                tensor.cuda().view(-1, args.seq_len, args.input_dim) for tensor in batch
            ]
            (
                info,
                mask,
                real,
            ) = batch
            batch_size = real.size(0)
            for _ in range(num_samples):
                t += 1
                noise = (
                    torch.randn(batch_size, args.noise_dim)
                    .cuda()
                    .view(-1, args.noise_dim, 1)
                )
                fake = generator(noise, info)
                fake_input = fake.view(-1, args.seq_len, args.input_dim)
                fake_input[mask.bool()] = real[mask.bool()]
                mean_input = filling(info, mask, method="mean")
                bfill_input = filling(info, mask, method="bfill")
                if t % 50 == 0:
                    plot_evaluation(
                        args,
                        fakes_dict={
                            "mean": mean_input[0].view(-1).tolist(),
                            "bfill": bfill_input[0].view(-1).tolist(),
                            "gan": fake_input[0].view(-1).tolist(),
                        },
                        real=real[0].view(-1).tolist(),
                        mask=mask[0].view(-1).tolist(),
                        iters=t,
                        path=path,
                    )
                MAE["fake"].append(mae(fake_input.detach().cpu(), real.detach().cpu()))
                MAE["mean"].append(mae(mean_input.detach().cpu(), real.detach().cpu()))
                MAE["bfill"].append(
                    mae(bfill_input.detach().cpu(), real.detach().cpu())
                )
            pbar.update(1)
    return MAE


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="checkpoints", type=str)
parser.add_argument("--lstm_path", type=str)
parser.add_argument("--num_samples", default=20, type=int)


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [os.path.join(args.model_path, file_) for file_ in filenames]
    else:
        paths = [args.model_path]
    # if args.lstm_path is not None:
    #     for path in paths:
    #         checkpoint = torch.load(path)
    #         _args = AttrDict(checkpoint["args"])
    #         dset, loader = data_loader(_args, mode="test")

    #         lstm_checkpoint = torch.load(args.lstm_path)
    #         _lstm_args = AttrDict(lstm_checkpoint["args"])
    #         generator = get_generator(checkpoint)
    #         LSTM_model = get_lstm_1(lstm_checkpoint)

    #         print(
    #             "Seq Len: {}, Mask rate: {} (path={})".format(
    #                 _args.seq_len, _args.mask_rate, path
    #             )
    #         )
    #         for k, v in mapes.items():
    #             print("\tMape({}): {:.4f}".format(k, sum(v) / len(v)))
    for path in paths:
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint["args"])
        dset, loader = data_loader(_args, mode="test")
        maes = evaluate(_args, loader, generator, args.num_samples, path)
        print(
            "Seq Len: {}, Mask rate: {} (path={})".format(
                _args.seq_len, _args.mask_rate, path
            )
        )
        for k, v in maes.items():
            print("\tMape({}): {:.4f}".format(k, sum(v) / len(v)))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
