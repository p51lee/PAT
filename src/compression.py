import os
from utils import comp_data

sys_name = "3ptl_2dim_long"

comp_rate_list = [2 ** n for n in range(1, 9)]

'''
data_comp 디렉토리 안에 system name 디렉토리를 만들고,
그 안에 다시 001 이라는 이름의 디렉토리를 만들어서
원본 데이터를 넣어주어야 재귀적? 으로 압축이 가능하다.
'''

for comp_rate in comp_rate_list:
    file_index = 0
    while True:
        success = comp_data(sys_name, file_index, comp_rate)
        if not success:
            break
        print("compresson rate: {:06d} | ".format(comp_rate),
              "file index: {:09d}".format(file_index))
        file_index += 1
