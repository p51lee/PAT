import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from utils import make_batch, load_data
from models import FCGAT

"""
2dim 3ptl lin
    064: 86th, 0.0365
    128: 66th, 0.1452
    256: 60th, 0.5293
"""

sys_name = "3ptl_2dim_lin"
comp_rate = 128
sys_comp_name = sys_name + "_" + "{:04d}".format(comp_rate)
best_epoch = 66
dt = 0.0005 * comp_rate

dimension = 2
num_particle = 3
hidden1 = 256
hidden2 = 128
dropout = 0.2
nb_heads1 = 16
nb_heads2 = 8
alpha = 0.01

model = FCGAT(n_input_features=dimension,
              n_hidden_features1=hidden1,
              n_hidden_features2=hidden2,
              n_output_features=dimension,
              dropout=dropout,
              n_heads1=nb_heads1,
              n_heads2=nb_heads2,
              alpha=alpha,
              num_particle=num_particle
              )
model.eval()

def step(init_frame, time_interval):
    init_state_pos = [ptl_state[0:2] for ptl_state in init_frame]
    next_state_pos = []

    for ptl_idx in range(num_particle):
        init_frame_rev = init_frame[ptl_idx:] + init_frame[:ptl_idx]
        input_chars = []
        for index_ps, ptl_state in enumerate(init_frame_rev):
            if index_ps == 0:
                input_chars.append(init_frame_rev[index_ps][2:4]) # 속도넣기
            else:
                input_chars.append(init_frame_rev[index_ps][0:2]) # 위치넣기
        next_state_pos.append(model(input_chars))

    next_state_vel = [[(next_state_pos[idx_ptl][k]-init_state_pos[idx_ptl][k]) / time_interval for k in range(dimension)] for idx_ptl in range(num_particle)]
    next_frame = [next_state_pos[idx_ptl] + next_state_vel[idx_ptl] for idx_ptl in range(num_particle)]

    return next_frame


# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('../model_save/{0}_epoch{1:05d}.pkl'.format(sys_comp_name, best_epoch)))

file_index = 0
while True:
    dir = "../data_prediction/" + sys_comp_name
    if not os.path.exists(dir):
        os.makedirs(dir)

    fd = open("../data_prediction/{0}/{1}.txt".format(sys_comp_name, str(file_index).zfill(10)))
    fd.write("{0}\n{1}\n".format(dimension, num_particle))

    data = load_data(sys_comp_name, file_index)
    if not data:
        break

    n_frames = len(data)
    initial_frame = data[0]

    for _ in range(n_frames):
        next_frame = step(initial_frame, dt)
        for sth_to_write in sum(next_frame, []):
            fd.write(str(sth_to_write) + "\n")
    fd.close()
    file_index += 1