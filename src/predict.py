import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from utils import make_batch, load_data, make_batch_rev
from models import FCGAT, RPATRecursive

"""
2dim 3ptl lin
    064: 86th, 0.0365 무효
    128: 35th, 0.1652
    256: 8th, 0.8928
    512: 5th, 3.7892
    
2dim 3ptl lin with acc
    512 04th, 1.8694
    256 34th, 0.2737
    128 34th, 0.0691
    064 23th, 0.0190
    032 21th, 0.0043
    016 35th, 0.0004
    008 29th, 0.0004
    
    2dim 3ptl
    256 52nd, 0.7872
"""

sys_name = "3ptl_2dim"
comp_rate = 256
sys_comp_name = sys_name + "_" + "{:06d}".format(comp_rate)
best_epoch = 52
dt = 2**(-14) * comp_rate

dimension = 2
num_particle = 3
hidden1 = 256
# hidden2 = 128
dropout = 0.1
nb_heads1 = 8
# nb_heads2 = 8
alpha = 0.2

model = RPATRecursive(
    num_particles=num_particle,
    dimension=dimension,
    n_hidden_features=hidden1,
    dropout=dropout,
    alpha=alpha,
    n_heads=nb_heads1
)
model.eval()



# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('../model_save/{0}_epoch{1:05d}.pkl'.format(sys_comp_name, best_epoch)))

# TODO: 일단 임시로 해놓음
sys_comp_name = "3ptl_2dim_long" + "_" + "{:06d}".format(comp_rate)

file_index = 0
while True:
    print("file {} start".format(file_index))

    dir = "../data_prediction/" + sys_comp_name
    if not os.path.exists(dir):
        os.makedirs(dir)

    fd = open("../data_prediction/{0}/{1}.txt".format(sys_comp_name, str(file_index).zfill(10)), 'w')
    fd.write("{0}\n{1}\n".format(dimension, num_particle))

    data = load_data(sys_comp_name, file_index)
    if not data:
        break
    else:
        data = torch.FloatTensor(data)

    input_features_batch, _ = make_batch_rev(data)

    input_features_batch = input_features_batch

    output_batch = model(input_features_batch)

    for frame in output_batch:
        for sth_to_write in sum(frame.tolist(), []):
            fd.write(str(sth_to_write) + "\n")
    fd.close()
    file_index += 1
