import numpy as np
import pandas as pd
import torch
import torch.utils.data as td
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Split time series data given window size
def tseries_train_test_split(data, test_size, window_size):
    T = []
    for i in range(len(data) - window_size + 1):
        T.append(data[i : i + window_size, -1])
    wall = int(len(T) * (1 - test_size))
    train_data = np.array(T[:wall], dtype=np.float64)
    test_data = np.array(T[wall:], dtype=np.float64)
    return train_data, test_data


# preprocess data
def process_data(
    path,
    window_size,
    test_size,
    start_date: str = "2018-12-25",
    end_date: str = "2019-12-25",
):
    df = pd.read_csv(path, skiprows=[0])
    df = df.filter(items=["Date", "Close"])
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
        2 * (df["Close"] - df["Lower Band"]) / (df["Upper Band"] - df["Lower Band"]) - 1
    )
    #
    df = df.filter(["Date", "Symbol", "MinMaxed Close"])
    # log scaling
    ## df["Close"] = np.log10(df["Close"])
    train_data, test_data = tseries_train_test_split(
        df.values, test_size=test_size, window_size=window_size
    )

    return train_data, test_data


def mask_data(input: torch.Tensor, mask_where="random", mask_prob=0.3):
    if mask_where == "random":
        masks = torch.bernoulli(torch.full(input.shape, 1 - mask_prob))
        return masks
    elif mask_where == "end":
        seq_len = input.shape[0]
        masks = torch.ones_like(input)
        masks[int(seq_len * (1 - mask_prob)) :] = 0
        return masks
    raise ValueError(
        "Mask method should be random or end, but got {}".format(mask_where)
    )


class CrytoDataset(td.Dataset):
    def __init__(
        self,
        path: str,
        model_name: str,
        mode="train",
        test_size=0.2,
        window_size=50,
        mask_prob=0.4,
        mask_where="random",
    ):
        self.model_name = model_name
        assert mode in ["train", "test"]
        if mode == "test":
            torch.manual_seed(880910)
        self.train_data, self.test_data = process_data(
            path, test_size=test_size, window_size=window_size
        )
        # if model_name == "lstm":
        # self.scaler = StandardScaler()
        # self.scaler.fit(self.train_data)
        # self.train_data = self.scaler.transform(self.train_data)
        # self.test_data = self.scaler.transform(self.test_data)
        #
        self.inputs = self.train_data if mode == "train" else self.test_data
        self.inputs = torch.Tensor(self.inputs)
        self.mask = mask_data(
            self.inputs[0], mask_prob=mask_prob, mask_where=mask_where
        )
        self.labels = self.inputs.clone()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = self.inputs[idx]
        label = self.labels[idx]
        mask = self.mask
        input[~mask.bool()] = input[mask.bool()].sum() / mask.sum()
        # if self.model_name == "cond-dcgan":
        #     scaler = MinMaxScaler().fit(input.reshape(-1, 1))
        #     label = scaler.transform(label.reshape(-1, 1)).reshape(-1)
        #     input = scaler.transform(input.reshape(-1, 1)).reshape(-1)
        return (
            torch.FloatTensor(input),
            torch.FloatTensor(mask),
            torch.FloatTensor(label),
        )


def data_loader(args, mode):
    dset = CrytoDataset(
        path=args.datapath,
        model_name=args.model,
        mode=mode,
        test_size=args.test_size,
        window_size=args.seq_len,
        mask_prob=args.mask_rate,
        mask_where=args.mask_where,
    )
    if mode == "train":
        dloader = td.DataLoader(dset, batch_size=args.batch_size, shuffle=True)
    elif mode == "test":
        dloader = td.DataLoader(dset, batch_size=args.batch_size, shuffle=False)

    return dset, dloader
