import torch
import os
from utils import load_data


x = torch.FloatTensor([[1,2],[3,4]])
y = torch.FloatTensor([[2,2],[2,2]])
print(x)
print(torch.mean(x, 0))