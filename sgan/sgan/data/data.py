import pandas as pd
import os
import logging
import sys
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
import math
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler

FORMAT = "[%(levelname)s] %(filename)s | %(lineno)4d | %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj

    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    if res_x >= threshold:
        return 1.0
    else:
        return 0.0


def read_file(
    data_dir: str, start_date: str = "2018-12-25", end_date: str = "2019-12-25"
) -> List[np.ndarray]:
    """_summary_

    Args:
        data_dir (str): data directory
        start_date (str, optional): start date. Defaults to "2018-12-25".
        end_date (str, optional): end date. Defaults to "2022-12-25".

    Returns:
        List[np.ndarray]: data array (#frames, #symbols, #features)
    """
    all_files = [_path for _path in os.listdir(data_dir) if _path.endswith(".csv")]
    token_list = [_path.split("_")[1] for _path in all_files]
    token_le = LabelEncoder()
    time_le = LabelEncoder()
    # scaler = StandardScaler()
    token_le.fit(token_list)

    all_files = [os.path.join(data_dir, _path) for _path in all_files]
    logger.info("All tokens: {}".format(token_list))
    data_list = []

    for idx, file in enumerate(all_files):
        df = pd.read_csv(file, skiprows=[0])
        df = df.filter(["Date", "Symbol", "Close"])
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[(df.Date >= start_date) & (df.Date <= end_date)]
        df = df.sort_values("Date", ascending=True)
        # [Normalize] with rolling avg/std
        df["MA30"] = df["Close"].rolling(30).mean()
        df["MS30"] = df["Close"].rolling(30).std()
        df = df.dropna()
        df["Upper Band"] = df["MA30"] + (df["MS30"] * 4)
        df["Lower Band"] = df["MA30"] - (df["MS30"] * 4)
        df["Normalized Close"] = (df["Close"] - df["MA30"]) / df["MS30"]
        df["MinMaxed Close"] = (
            2 * (df["Close"] - df["Lower Band"]) / (df["Upper Band"] - df["Lower Band"])
            - 1
        )
        #
        n_row = df["Close"].count()
        logger.info(
            "[{}] start={} end={} n_rows={}".format(
                token_list[idx], df["Date"].min(), df["Date"].max(), n_row
            )
        )
        if idx == 0:
            time_le.fit(df["Date"])
        df["Date"] = time_le.transform(df["Date"])
        df["Symbol"] = token_le.transform(df["Symbol"])
        df = df.filter(["Date", "Symbol", "MinMaxed Close"])
        data_list.append(df.values)
    data_list = np.asarray(data_list, dtype=np.float64)
    data_list = data_list.transpose((1, 0, 2))

    # NOTE: normalization might effect MAPE performance
    # data_list[:, :, 2] = scaler.fit_transform(data_list[:, :, 2])
    # data_list[:, :, 2] = np.log2(data_list[:, :, 2])
    return data_list


class CryptoDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
        self,
        data_dir,
        obs_len=8,
        pred_len=12,
        skip=1,
        threshold=0.002,
        min_ped=1,
        delim="\t",
        mode="train",
        testsize=0.4,
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(CryptoDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        frame_data = read_file(data_dir=self.data_dir)
        if mode == "train":
            frame_data = frame_data[: int(-testsize * len(frame_data))]
        elif mode == "val":
            frame_data = frame_data[int(-testsize * len(frame_data)) :]
        else:
            ValueError("mode must be train or val")
        logger.info("Read data with shape = {}".format(frame_data.shape))

        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []

        frames = np.unique(frame_data[:, 0, 0]).tolist()
        peds_in_curr_seq = np.unique(frame_data[0, :, 1])
        num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))
        for idx in tqdm(range(0, num_sequences * self.skip + 1, skip)):
            curr_seq_data = np.concatenate(frame_data[idx : idx + self.seq_len], axis=0)
            curr_seq_rel = np.zeros((len(peds_in_curr_seq), 1, self.seq_len))
            curr_seq = np.zeros((len(peds_in_curr_seq), 1, self.seq_len))
            curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
            num_peds_considered = 0
            _non_linear_ped = []
            for _, ped_id in enumerate(peds_in_curr_seq):
                curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                if pad_end - pad_front != self.seq_len:
                    continue
                curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                curr_ped_seq = curr_ped_seq
                # Make coordinates relative
                rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                _idx = num_peds_considered
                curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                # Linear vs Non-Linear Trajectory
                _non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))
                curr_loss_mask[_idx, pad_front:pad_end] = 1
                num_peds_considered += 1

            if num_peds_considered > min_ped:
                non_linear_ped += _non_linear_ped
                num_peds_in_seq.append(num_peds_considered)
                loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                seq_list.append(curr_seq[:num_peds_considered])
                seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)
        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(seq_list[:, :, : self.obs_len]).type(
            torch.float
        )
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len :]).type(
            torch.float
        )
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, : self.obs_len]).type(
            torch.float
        )
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len :]).type(
            torch.float
        )
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :],
            self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :],
            self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end],
            self.loss_mask[start:end, :],
        ]
        return out


if __name__ == "__main__":
    dset = CryptoDataset(data_dir="sgan/data")
