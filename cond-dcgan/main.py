import torch

a = torch.ones(10, 5)
b = torch.zeros(10, 1)
a = torch.concat((a, b), dim=1)
print(a)
a = a[:, 1:]
print(a)
