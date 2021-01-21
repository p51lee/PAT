import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from utils import make_batch, load_data
from models import FCGAT

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
"""

sys_name = "3ptl_2dim_lin"
comp_rate = 512
sys_comp_name = sys_name + "_" + "{:03d}".format(comp_rate)
best_epoch = 4
dt = 0.0005 * comp_rate

dimension = 2
num_particle = 3
hidden1 = 256
hidden2 = 128
dropout = 0.2
nb_heads1 = 16
nb_heads2 = 8
alpha = 0.01

model = FCGAT(n_input_features=dimension*2,
              n_hidden_features1=hidden1,
              n_hidden_features2=hidden2,
              n_output_features=dimension*3,
              dropout=dropout,
              n_heads1=nb_heads1,
              n_heads2=nb_heads2,
              alpha=alpha,
              num_particle=num_particle
              )
model.eval()

def step_old(init_frame, time_interval): # unused
    init_state_pos = torch.FloatTensor([ptl_state[0:2] for ptl_state in init_frame])
    diff_state_pos = []

    for ptl_idx in range(num_particle):
        init_frame_rev = init_frame[ptl_idx:] + init_frame[:ptl_idx]
        input_chars = []
        for index_ps, ptl_state in enumerate(init_frame_rev):
            if index_ps == 0:
                input_chars.append(init_frame_rev[index_ps][2:4]) # 속도넣기
            else:
                rel_position = np.array(init_frame_rev[index_ps][0:2]) - np.array(init_frame_rev[0][0:2])
                input_chars.append(rel_position.tolist()) # 위치넣기

        input_chars = torch.FloatTensor(input_chars)
        diff_chars = model(input_chars)
        # print(diff_chars)
        # print('hey', input_chars, '\n', diff_chars)
        diff_state_pos.append(diff_chars.tolist())

    next_state_pos = init_state_pos + torch.FloatTensor(diff_state_pos)

    # next_state_vel = [[(next_state_pos[idx_ptl][k]-init_state_pos[idx_ptl][k]) / time_interval for k in range(dimension)] for idx_ptl in range(num_particle)]
    next_state_vel = (torch.FloatTensor(diff_state_pos)/time_interval)
    # print(next_state_vel)
    next_state_pos = next_state_pos
    # next_frame = [next_state_pos[idx_ptl] + next_state_vel[idx_ptl] for idx_ptl in range(num_particle)]
    next_frame = torch.cat((next_state_pos, next_state_vel), 1).tolist()

    return next_frame

def step(init_frame_1st, init_frame_2nd, time_interval):
    init_state_pos = torch.FloatTensor([ptl_state[0:2] for ptl_state in init_frame_2nd])
    init_state_vel = torch.FloatTensor([ptl_state[2:4] for ptl_state in init_frame_2nd])
    diff_state_pos = []
    diff_state_vel = []

    for ptl_idx in range(num_particle):
        init_frame_rev_1st = init_frame_1st[ptl_idx:] + init_frame_1st[:ptl_idx]
        init_frame_rev_2nd = init_frame_2nd[ptl_idx:] + init_frame_2nd[:ptl_idx]
        input_chars = []
        for index_ps, ptl_state in enumerate(init_frame_rev_2nd):
            if index_ps == 0:
                delta_v = np.array(init_frame_rev_2nd[index_ps][2:4]) - np.array(init_frame_rev_1st[index_ps][2:4])
                input_chars.append(init_frame_rev_2nd[index_ps][2:4] + delta_v.tolist()) # 속도넣기 추가된 속도변화넣기
            else:
                rel_position = np.array(init_frame_rev_2nd[index_ps][0:2]) - np.array(init_frame_rev_2nd[0][0:2])
                rel_velosity = np.array(init_frame_rev_2nd[index_ps][2:4]) - np.array(init_frame_rev_2nd[0][2:4])
                input_chars.append(rel_position.tolist() + rel_velosity.tolist()) # 위치넣기와 추가된 속도넣기

        input_chars = torch.FloatTensor(input_chars)
        diff_chars = model(input_chars)
        # print(diff_chars)
        # print('hey', input_chars, '\n', diff_chars)
        diff_state_pos.append(diff_chars.tolist()[0:2])
        diff_state_vel.append(diff_chars.tolist()[2:4])

    next_state_pos = init_state_pos + torch.FloatTensor(diff_state_pos)
    next_state_vel = init_state_vel + torch.FloatTensor(diff_state_vel)

    # next_state_vel = [[(next_state_pos[idx_ptl][k]-init_state_pos[idx_ptl][k]) / time_interval for k in range(dimension)] for idx_ptl in range(num_particle)]
    # print(next_state_vel)
    # next_frame = [next_state_pos[idx_ptl] + next_state_vel[idx_ptl] for idx_ptl in range(num_particle)]
    next_frame = torch.cat((next_state_pos, next_state_vel), 1).tolist()

    return next_frame

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('../model_save/{0}_epoch{1:05d}.pkl'.format(sys_comp_name, best_epoch)))

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

    n_frames = len(data)
    initial_frame_1st = data[0]
    initial_frame_2nd = data[1]

    for _ in range(n_frames):
        next_frame = step(initial_frame_1st, initial_frame_2nd, dt)
        for sth_to_write in sum(next_frame, []):
            fd.write(str(sth_to_write) + "\n")
        initial_frame_1st = initial_frame_2nd
        initial_frame_2nd = next_frame
    fd.close()
    file_index += 1