import matplotlib.pyplot as plt
from celluloid import Camera
import os

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

system_name = '3dim_2ptl' # input("Enter system name")

fig, ax = plt.subplots ()
camera = Camera (fig)

raw_data = load_data (system_name, 0)
for one_frame in raw_data:
    for particle in one_frame:
        ax.scatter (particle[0], particle[1], c='blue', s=200)
    camera.snap()

animation = camera.animate(interval=64, repeat = True)
animation.save ('test.mp4')