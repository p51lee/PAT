from particle import IdenticalParticle
from system import System

import numpy as np
import matplotlib.pyplot as plt

dim = 2
n_particle = 3
dt = 2 ** (-14)

system_2 = System(name="2pinned_8f_16fps",
                  n=-1.,
                  k=-1.,
                  dt=dt,
                  n_particles=n_particle,
                  dim=dim,
                  save=True)

# 시스템에 particle class 들을 초기화시켜서 넣어준다

# 첫 번째 움직일 수 있는 particle
system_2.add(IdenticalParticle(m=1,
                               x_init=np.array([0] * dim, dtype=float),
                               v_init=np.array([0] * dim, dtype=float),
                               pinned=False))

# 나머지 고정된 particle 들
for _ in range(n_particle - 1):
    ptl = IdenticalParticle(m=1,
                            x_init=np.array([0] * dim, dtype=float),
                            v_init=np.array([0] * dim, dtype=float),
                            pinned=True)
    system_2.add(ptl)

system_2.make_testcase(frame_number=2 ** 13, testcase_number=400, min_distance=0.00001)
# system_2.make_testcase(frame_number=65536, testcase_number=1, min_distance=0.00001)

# system_2.make_testcase(200, 5, 0.01)
# n = 1000000
# for i in range(n):
#     if (100 * i) % n == 0:
#         print ((100 * i) / n, "%", end='')
#     system_2.step()

print(system_2)
