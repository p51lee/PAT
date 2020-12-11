import os
from utils import comp_data

sys_name = "3ptl_2dim_lin"


comp_rate_list = [256, 512, 1024] #
# comp_rate_list = [256, 512, 1024]
comp_rate_list = [2**n for n in range(1,14)]
for comp_rate in comp_rate_list:
    file_index = 0
    while True:
        success = comp_data(sys_name, file_index, comp_rate)
        if not success:
            break
        print("compresson rate: {:04d} | ".format(comp_rate),
              "file index: {:04d}".format(file_index))
        file_index += 1