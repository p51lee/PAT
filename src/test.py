import torch

t = torch.tensor([[1], [2], [3]])
a, b = t.size()
print(torch.cuda.is_available())