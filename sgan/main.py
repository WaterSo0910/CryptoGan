import torch

a = [1, 2, 1, 3]
b = [1, 1, 1, 0]
ta = torch.tensor(a)
tb = torch.tensor(b)
print(ta[tb.bool()].size() == torch.Size([3]))
