import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc

from system import System


class Visualizer():
    def __init__(self, sys_name):
        self.name = sys_name

        file_name = "./{0}.txt".format(sys_name)
        fd = open(file_name, 'r')
        temp = fd.readlines()
        self.data_raw = [float(e.strip()) for e in temp]
        fd.close()

        self.dim = int(self.data_raw[0])
        self.n_particles = int(self.data_raw[1])
        self.data = []

        for i in range(self.n_particles):
            m = self.data_raw[2 + i * (self.dim + 1)]
            x = self.data_raw[2 + i * (self.dim + 1) + 1: 2 + (i + 1) * (self.dim + 1)]
            self.data.append((m, x))


    def load_from_file(self):
        pass

    def plot(self):

        pass

vis = Visualizer("2dim2ptl")