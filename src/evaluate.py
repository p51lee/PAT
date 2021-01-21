import torch
import torch.nn as nn
import numpy as np

from utils import load_prediction, load_data

option = 2 # 1 for PAT prediction, 2 for GRAVITY prediction
comp_rate = 8
time_interval = 0.0005 * comp_rate
sys_name = "3ptl_2dim_lin"
sys_comp_name = sys_name + "_" + "{:03d}".format(comp_rate)

criterion = nn.L1Loss()

file_index = 0
if option == 1: # PAT prediction
    losses_per_each_file = []
    while True:
        data = load_prediction(sys_comp_name, file_index)
        data_True = load_data(sys_comp_name, file_index)

        if not(data_True and data):
            break


        data_position = [[ptl_char[0:2] for ptl_char in frame] for frame in data[:max((len(data))//500, 30)]]
        data_True_position = [[ptl_char[0:2] for ptl_char in frame] for frame in data_True[:max((len(data))//500, 30)]]


        data_position = torch.FloatTensor(data_position)
        data_True_position = torch.FloatTensor(data_True_position)

        # losses = [criterion(x_predict, x_true) for x_predict, x_true in zip(data_position, data_True_position)]
        loss = criterion(data_position, data_True_position)

        print("file index: {0}     loss: {1}".format(file_index, loss))

        losses_per_each_file.append(loss)
        file_index += 1


    losses_per_each_file_torch = torch.FloatTensor(losses_per_each_file)
    mean_losses_per_each_step = torch.mean(losses_per_each_file_torch, 0)
    mean_losses_per_time = mean_losses_per_each_step / time_interval

    print("mean losses per time: {}".format(mean_losses_per_time))


    # fd = open('../losses/{0}_{1:03d}_PAT'.format(sys_name, comp_rate))
    # fd.write("PAt prediction Loss:\n{}\n".format(mean_losses_per_time.data))


elif option == 2: # GRAVITY prediction
    sys_gravity_name = sys_name + "_control_{:03d}".format(comp_rate)
    losses_per_each_file = []
    while True:
        data = load_data(sys_gravity_name, file_index)
        data_True = load_data(sys_comp_name, file_index)

        if not(data_True and data):
            break

        data_position = [[ptl_char[0:2] for ptl_char in frame] for frame in data[:max((len(data))//5, 10)]]
        data_True_position = [[ptl_char[0:2] for ptl_char in frame] for frame in data_True[:max((len(data))//5, 10)]]

        data_position = torch.FloatTensor(data_position)
        data_True_position = torch.FloatTensor(data_True_position)

        # losses = [criterion(x_predict, x_true) for x_predict, x_true in zip(data_position, data_True_position)]
        loss = criterion(data_position, data_True_position)

        print("file index: {0}     loss: {1}".format(file_index, loss))

        losses_per_each_file.append(loss)
        file_index += 1

    losses_per_each_file_torch = torch.FloatTensor(losses_per_each_file)
    print(losses_per_each_file_torch)
    mean_losses_per_each_step = torch.mean(losses_per_each_file_torch, 0)
    mean_losses_per_time = mean_losses_per_each_step / time_interval

    print("mean losses per time: {}".format(mean_losses_per_time))

    # fd = open('../losses/{0}_{1:04d}_PAT'.format(sys_name, comp_rate))
    # for meanloss in mean_losses_per_each_step:
    #     fd.write("Classical Prediction Loss:\n{}\n".format(meanloss))
