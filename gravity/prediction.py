from particle import IdenticalParticle
from system import System
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/')))

from gravity_utils import load_data

sys_origin = "3ptl_2dim_lin"
sys_name = sys_origin + "_control"
comp_rate = 128
sys_comp_name = sys_name + "_" + "{:03d}".format(comp_rate)
sys_origin_comp = sys_origin +  "_" + "{:03d}".format(comp_rate)

dim = 2
n_particle = 3
dt = 0.0005 * comp_rate

dir = "../data/" + sys_comp_name
if not os.path.exists(dir):
    os.makedirs(dir)

file_index = 0
while True:
    data = load_data(sys_origin_comp, file_index)
    if not data:
        break

    n_frames = len(data)
    initial_frame = data[0]

    system_2 = System(name=sys_comp_name,
                      n=-1.,
                      k=-1.,
                      dt=dt,
                      n_particles=n_particle,
                      dim=dim,
                      save=True)

    for ptl_idx in range(n_particle):
        ptl1 = IdenticalParticle(m=1,
                                 x_init=np.array(initial_frame[ptl_idx][0:2], dtype=float),
                                 v_init=np.array(initial_frame[ptl_idx][2:4], dtype=float))
        system_2.add(ptl1)

    system_2.file_init(file_index)
    for _ in range(n_frames):
        system_2.step(file_index)

    file_index += 1

for _ in range(n_particle):
    ptl1 = IdenticalParticle(m=1,
                             x_init=np.array([0] * dim, dtype=float),
                             v_init=np.array([0] * dim, dtype=float))
    system_2.add(ptl1)