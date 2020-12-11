from particle import IdenticalParticle
from system import System

import numpy as np
import matplotlib.pyplot as plt

dim = 2
n_particle = 3
dt = 0.0005

system_2 = System(name = "3ptl_2dim_lin",
                  n = -1.,
                  k = -1.,
                  dt = dt,
                  n_particles = n_particle,
                  dim = dim,
                  save = True)

for _ in range(n_particle):
    ptl1 = IdenticalParticle(m=1,
                             x_init=np.array([0] * dim, dtype=float),
                             v_init=np.array([0] * dim, dtype=float))
    system_2.add(ptl1)

system_2.make_testcase(5000, 200, 0.01)
# system_2.make_testcase(200, 5, 0.01)
# n = 1000000
# for i in range(n):
#     if (100 * i) % n == 0:
#         print ((100 * i) / n, "%", end='')
#     system_2.step()

print(system_2)