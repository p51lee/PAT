import torch

t = torch.tensor([[1], [2], [3]])
a, b = t.size()
print(t[0:2])