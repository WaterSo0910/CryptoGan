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
def process_data(path, window_size, test_size):
    df = pd.read_csv(path, skiprows=[0])
    df = df.filter(items=["Date", "Close"])
    df["Date"] = pd.to_datetime(df["Date"])
    # df["MA30"] = df["Close"].rolling(30).mean()
    # df["Close"] = df["Close"] - df["MA30"]
    # df = df.dropna()

    # log scaling
    ## df["Close"] = np.log10(df["Close"])
    train_data, test_data = tseries_train_test_split(
        df.values, test_size=test_size, window_size=window_size
    )

    return train_data, test_data


def mask_data(inputs: torch.Tensor, mask_where="random", mask_prob=0.3):
    if mask_where == "random":
        masks = torch.bernoulli(torch.full(inputs.shape, 1 - mask_prob))
        return masks
    elif mask_where == "end":
        seq_len = inputs.shape[1]
        masks = torch.ones_like(inputs)
        masks[:, int(seq_len * (1 - mask_prob)) :] = 0
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
        self.train_data, self.test_data = process_data(
            path, test_size=test_size, window_size=window_size
        )
        if model_name == "lstm":
            self.scaler = StandardScaler()
            self.scaler.fit(self.train_data)
            self.train_data = self.scaler.transform(self.train_data)
            self.test_data = self.scaler.transform(self.test_data)
        self.inputs = self.train_data if mode == "train" else self.test_data
        self.inputs = torch.Tensor(self.inputs)
        self.masks = mask_data(self.inputs, mask_prob=mask_prob, mask_where=mask_where)
        self.labels = self.inputs.clone()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = self.inputs[idx]
        label = self.labels[idx]
        mask = self.masks[idx]

        input[~mask.bool()] = input[mask.bool()].sum() / mask.sum()
        if self.model_name == "cond-dcgan":
            scaler = MinMaxScaler().fit(input.reshape(-1, 1))
            label = scaler.transform(label.reshape(-1, 1)).reshape(-1)
            input = scaler.transform(input.reshape(-1, 1)).reshape(-1)
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
