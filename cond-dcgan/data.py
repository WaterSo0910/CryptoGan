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

    df["Close"] = np.log10(df["Close"])
    # log scaling
    train_data, test_data = tseries_train_test_split(
        df.values, test_size=test_size, window_size=window_size
    )
    return train_data, test_data


def mask_data(inputs: torch.Tensor, mask_prob=0.3):
    masks = torch.bernoulli(torch.full(inputs.shape, 1 - mask_prob))
    return masks


class CrytoDataset(td.Dataset):
    def __init__(
        self, path: str, mode="train", test_size=0.2, window_size=50, mask_prob=0.4
    ):
        assert mode in ["train", "test"]
        self.train_data, self.test_data = process_data(
            path, test_size=test_size, window_size=window_size
        )
        self.inputs = self.train_data if mode == "train" else self.test_data
        self.inputs = torch.Tensor(self.inputs)
        self.masks = mask_data(self.inputs, mask_prob=mask_prob)
        self.labels = self.inputs.clone()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(
            self.labels[idx].reshape(-1, 1)
        )
        label = scaler.transform(self.labels[idx].reshape(-1, 1)).reshape(-1)
        input = scaler.transform(self.inputs[idx].reshape(-1, 1)).reshape(-1)
        input = torch.FloatTensor(input)
        mask = self.masks[idx]
        s = torch.randn(input.size())
        input[~mask.bool()] = s[~mask.bool()]
        return (
            torch.FloatTensor(input),
            torch.FloatTensor(mask),
            torch.FloatTensor(label),
        )
