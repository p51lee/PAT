from particle import IdenticalParticle
from system import System

import numpy as np
import matplotlib.pyplot as plt

dim = 2
n_particle = 3
dt = 0.1

system_2 = System(name="3ptl_2dim_debug",
                  n=-1.,
                  k=-1.,
                  dt=dt,
                  n_particles=n_particle,
                  dim=dim,
                  save=True)

# 시스템에 particle class 들을 초기화시켜서 넣어준다
for _ in range(n_particle):
    ptl = IdenticalParticle(m=1,
                             x_init=np.array([0] * dim, dtype=float),
                             v_init=np.array([0] * dim, dtype=float))
    system_2.add(ptl)

system_2.make_testcase(100, 5, 0.0001)
# system_2.make_testcase(200, 5, 0.01)
# n = 1000000
# for i in range(n):
#     if (100 * i) % n == 0:
#         print ((100 * i) / n, "%", end='')
#     system_2.step()

print(system_2)
