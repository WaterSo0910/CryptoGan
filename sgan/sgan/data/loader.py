from torch.utils.data import DataLoader

from sgan.data.trajectories import TrajectoryDataset, seq_collate
from sgan.data.data import CryptoDataset


def data_loader(args, mode):
    dset = CryptoDataset(
        data_dir=args.datapath,
        mode=mode,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim,
        testsize=args.testsize,
    )

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate,
    )
    return dset, loader
