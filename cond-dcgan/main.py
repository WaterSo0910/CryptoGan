import torch

a = torch.randn((6, 4, 3))
print(a)
b = a.roll(dims=(2), shifts=(1))
print(b)
