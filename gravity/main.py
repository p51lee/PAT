from particle import IdenticalParticle
from system import System

import numpy as np
import matplotlib.pyplot as plt

dim = 3
n_particle = 10
dt = 0.0001

system_2 = System(name = "10ptlgo",
                  n = 2.,
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

system_2.make_testcase(10, 1000, 0.01)
# n = 1000000
# for i in range(n):
#     if (100 * i) % n == 0:
#         print ((100 * i) / n, "%", end='')
#     system_2.step()

print(system_2)