import torch
import os.path


def load_datum(sys_name, file_index):
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
        n_frames = len(data_raw)-2
        datum = []

        for i in range(n_particles):



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
    n_states, _ = states.size()
    input_batch = []
    target_batch = []

    for i, state in enumerate(states):
        if i == 0:
            continue
        input_batch.append(states[i - 1])  # 이 전의 state 를 input 으로 기록
        target_batch.append(state)  # 결과 state

    return input_batch, target_batch
