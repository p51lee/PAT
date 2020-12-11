import torch
import torch.nn as nn

from utils import load_prediction, load_data

option = 1 # 1 for PAT prediction, 0 for GRAVITY prediction
comp_rate = 128
sys_name = "3ptl_2dim_lin"

criterion = nn.L1Loss()

file_index = 0
if option == 1: # PAT prediction
    losses_per_each_file = []
    while True:
        data = load_prediction(sys_name, file_index)
        data_True = load_data(sys_name, file_index)

        if not(data_True and data):
            break

        data_position = [[ptl_char[0:2] for ptl_char in frame] for frame in data]
        data_True_position = [[ptl_char[0:2] for ptl_char in frame] for frame in data_True]

        losses = [criterion(x_predict, x_true) for x_predict, x_true in zip(data_position, data_True_position)]

        losses_per_each_file.append(losses)
        file_index += 1

    losses_per_each_file_torch = torch.FloatTensor(losses_per_each_file)
    mean_losses_per_each_step = torch.mean(losses_per_each_file_torch, 0)

    fd = open('../losses/{0}_{1:04d}_PAT'.format(sys_name, comp_rate))
    for meanloss in mean_losses_per_each_step:
        fd.write("{}\n".format(meanloss))


elif option == 2: # GRAVITY prediction
    sys_gravity_name = sys_name + "_control"
    losses_per_each_file = []
    while True:
        data = load_data(sys_gravity_name, file_index)
        data_True = load_data(sys_name, file_index)

        if not(data_True and data):
            break

        data_position = [[ptl_char[0:2] for ptl_char in frame] for frame in data]
        data_True_position = [[ptl_char[0:2] for ptl_char in frame] for frame in data_True]

        losses = [criterion(x_predict, x_true) for x_predict, x_true in zip(data_position, data_True_position)]

        losses_per_each_file.append(losses)
        file_index += 1

    losses_per_each_file_torch = torch.FloatTensor(losses_per_each_file)
    mean_losses_per_each_step = torch.mean(losses_per_each_file_torch, 0)

    fd = open('../losses/{0}_{1:04d}_PAT'.format(sys_name, comp_rate))
    for meanloss in mean_losses_per_each_step:
        fd.write("{}\n".format(meanloss))
