from particle import IdenticalParticle
from system import System

import numpy as np
import matplotlib.pyplot as plt

system_2 = System(name = "2ptlgo",
                  n = 2.,
                  k = -1.,
                  dt = 0.001,
                  n_particles = 2,
                  dim = 2,
                  save = True)

ptl1 = IdenticalParticle(m=1,
                         x_init=np.array([0, 1], dtype=float),
                         v_init=np.array([1, 0], dtype=float))
ptl2 = IdenticalParticle(m=1,
                         x_init=np.array([0, -1], dtype=float),
                         v_init=np.array([-1, 0], dtype=float))
system_2.add(ptl1)
system_2.add(ptl2)
system_2.make_testcase(10, 3, 0.01)
# n = 1000000
# for i in range(n):
#     if (100 * i) % n == 0:
#         print ((100 * i) / n, "%", end='')
#     system_2.step()

print(system_2)