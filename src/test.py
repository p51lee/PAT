import torch
import os
from utils import load_data


x = torch.FloatTensor([[0,0],[0,0]])
y = torch.FloatTensor([[2,2],[2,2]])
criterion = torch.nn.MSELoss()
print(criterion(x,y))