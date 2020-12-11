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
        # mass_list = data_raw[2:2+n_particles]

        if (len(data_raw) - 2) % (2 * dim): # 안 나누어떨어질 경우
            print("invalid file length: file no{1} of {0}".format(sys_name, file_index))
            return False
        n_frames = (len(data_raw) - 2) // (2 * dim * n_particles)
        data = []

        for frame_index in range(n_frames):
            datum = []
            for particle_index in range(n_particles):
                temp = []
                start = 2 + frame_index*(dim*2*n_particles) + particle_index*(dim*2)
                x = data_raw[start: start + dim]
                v = data_raw[start + dim: start + dim*2]
                temp += x
                temp += v
                datum.append(temp)
            data.append(datum)

        return data


def load_prediction(sys_name, file_index):
    filename_list = os.listdir('../data_prediction/{0}'.format(sys_name))
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
        # mass_list = data_raw[2:2+n_particles]

        if (len(data_raw) - 2) % (2 * dim): # 안 나누어떨어질 경우
            print("invalid file length: file no{1} of {0}".format(sys_name, file_index))
            return False
        n_frames = (len(data_raw) - 2) // (2 * dim * n_particles)
        data = []

        for frame_index in range(n_frames):
            datum = []
            for particle_index in range(n_particles):
                temp = []
                start = 2 + frame_index*(dim*2*n_particles) + particle_index*(dim*2)
                x = data_raw[start: start + dim]
                v = data_raw[start + dim: start + dim*2]
                temp += x
                temp += v
                datum.append(temp)
            data.append(datum)

        return data

def comp_data(sys_name, file_index, comp_rate):

    # filename_list = os.listdir('../data/{0}'.format(sys_name))
    filename_list = os.listdir('../data_comp/{0}/{1:03d}'.format(sys_name, comp_rate//2))

    if file_index >= len(filename_list):
        return False
    elif filename_list[file_index][-3:] != 'txt':
        return False
    else:
        filename = filename_list[file_index]
        # filepath = '../data/{0}/'.format(sys_name) + filename
        filepath = '../data_comp/{0}/{1:03d}/'.format(sys_name, comp_rate//2) + filename
        fd = open(filepath, 'r')
        temp = fd.readlines()
        data_raw = [float(e.strip()) for e in temp]
        fd.close()

        dim = int(data_raw[0])
        n_particles = int(data_raw[1])
        # mass_list = data_raw[2:2 + n_particles]

        if (len(data_raw) - 2) % (2 * dim): # 안 나누어떨어질 경우
            print("invalid file length: file no{1} of {0}".format(sys_name, file_index))
            return False

        n_frames = (len(data_raw) - 2) // (2 * dim * n_particles)
        data = []

        frame_index = 0
        while frame_index < n_frames:
            datum = []
            for particle_index in range(n_particles):
                temp = []
                start = 2 + frame_index * (dim * 2 * n_particles) + particle_index * (dim * 2)
                x = data_raw[start: start + dim]
                v = data_raw[start + dim: start + dim * 2]
                temp += x
                temp += v
                datum.append(temp)
            data.append(datum)
            # frame_index += comp_rate
            frame_index += 2

        dir1 = '../data_comp/{0}'.format(sys_name)
        dir2 = '../data_comp/{0}/{1:03d}'.format(sys_name, comp_rate)
        if not os.path.exists(dir1):
            os.makedirs(dir1)
        if not os.path.exists(dir2):
            os.makedirs(dir2)

        filepath_comp = '../data_comp/{0}/{1:03d}/'.format(sys_name, comp_rate) + filename

        str_write = ''
        str_write += "{0}\n{1}\n".format(dim, n_particles)

        # for mass in mass_list:
        #     str_write += str(mass) + '\n'

        for datum_comp in data:
            for ptl_state in datum_comp: # except its mass
                for xsNvs in ptl_state:
                    str_write += str(xsNvs) + '\n'

        fd_comp = open(filepath_comp, 'w')
        fd_comp.write(str_write)
        fd_comp.close()

    return True

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

        # 첫 번째 입자에 대한 상대위치를 적자
        prev_state = states[i-1]
        first_particle_position_vec = prev_state[0][0:2]

        # input 으로 들어가는건 첫 번째 입자의 속도, 그리고 나머지들의 상대 위치이다.
        input_frame = []
        input_frame.append(prev_state[0][2:4]) # 먼저 첫 번째 입자의 속도를 넣는다.

        for idx, ptl_state in enumerate(prev_state):
            if idx == 0: # 우리는 나머지 입자들을 원한다.
                continue
            else:
                rel_position = []
                for pos_i, position in enumerate(ptl_state[0:2]):
                    rel_position.append(position-first_particle_position_vec[pos_i])
                input_frame.append(rel_position)

        input_batch.append(input_frame)  # 이 전의 state 를 input 으로 기록
        target_batch.append([state[0][j] - states[i-1][0][j] for j in range(len(state[0]))]) # 첫 번째 particle 의 운동 변화을 학습한다.  # 결과 state, 맨 앞의 질량을 뺸다.

    return input_batch, target_batch