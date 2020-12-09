import torch
import os.path


def load_data(sys_name, file_index):
    filename_list = os.listdir('../data/{0}'.format(sys_name))
    if file_index >= len(filename_list):
        return False
    elif filename_list[file_index][-3:] != 'txt':
        return False
    else:
        filename = filename_list[file_index]
        filepath = '../data/{0}/'.format(sys_name) + filename
        fd = open(filepath, 'r')
        temp = fd.readlines()
        data_raw = [float(e.strip()) for e in temp]
        fd.close()

        dim = int(data_raw[0])
        n_particles = int(data_raw[1])
        mass_list = data_raw[2:2+n_particles]

        if (len(data_raw) - 2 - n_particles) % (2 * dim): # 안 나누어떨어질 경우
            print("invalid file length: file no{1} of {0}".format(sys_name, file_index))
            return False
        n_frames = (len(data_raw) - 2 - n_particles) // (2 * dim * n_particles)
        data = []

        for frame_index in range(n_frames):
            datum = []
            for particle_index in range(n_particles):
                temp = [mass_list[particle_index]]
                start = 2+n_particles + frame_index*(dim*2*n_particles) + particle_index*(dim*2)
                x = data_raw[start: start + dim]
                v = data_raw[start + dim: start + dim*2]
                temp += x
                temp += v
                datum.append(temp)
            data.append(datum)

        return data


# 한 번의 상황으로 여러 개의 학습 데이터를 만들 수 있다.?
def make_batch(states):
    """
    :param states:
    torch.tensor(
    [
    [state1], [state2], [state3], ...
    ]
    )
    :return:
    """
    n_states = len(states)
    input_batch = []
    target_batch = []

    for i, state in enumerate(states):
        if i == 0:
            continue
        input_batch.append(states[i - 1])  # 이 전의 state 를 input 으로 기록
        target_batch.append([ptl_state[1:] for ptl_state in state])  # 결과 state, 맨 앞의 질량을 뺸다.

    return input_batch, target_batch

