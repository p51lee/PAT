import numpy as np
import matplotlib.pyplot as plt
import os
import random

from typing import List

from particle import IdenticalParticle


class CentralForce():
    def __init__(self, n: float = 2., k: float = -1.):
        """F = {{k q1 q2} / {r^n}}"""
        self.n = n
        self.k = k
        self.n_particles = 0

    def f(self, q_src: float, x_src: np.ndarray, q_dst: float, x_dst: np.ndarray):
        if len(x_src) != len(x_dst):
            return False
        r_vec = x_dst - x_src
        r = np.sum(np.multiply(r_vec, r_vec)) ** 0.5
        r_unit = r_vec / r

        if (r == 0):
            # 두 입자의 위치가 같으면 그냥 0 array 을 리턴하자.
            return np.zeros_like(x_dst)

        return ( self.k * ( q_dst * q_src ) / (r ** self.n) ) * r_unit


class System():
    def __init__(self, name: str, n: float, k: float, dt: float, n_particles: int, dim = 2, save = True):
        self.name = name
        self.dt = dt  # time interval for each step
        self.cf = CentralForce(n, k)
        self.particles: List[IdenticalParticle] = []
        self.step_number = 0
        self.n_particles = n_particles  # 이만큼 particle 을 넣어야한다?
        self.n_particles_curr = 0
        self.save = save
        self.dim = dim

    def add(self, ptl: IdenticalParticle):
        # ptl.set_number(len(self.particles))  # starts from 0
        if self.dim != ptl.dim:
            return False
        self.particles.append(ptl)
        self.n_particles_curr += 1
        return True

    def step(self, testcase_index):
        if self.n_particles != self.n_particles_curr:
            # 아직 덜넣었다?
            print(f"Add {self.n_particles - self.n_particles_curr} more particles to run step().")
            return

        forces = []
        for ptl_dst in self.particles:
            force = np.zeros_like(ptl_dst.x)
            for ptl_src in self.particles:
                if ptl_dst == ptl_src:
                    continue
                else:
                    force += self.cf.f(ptl_src.m, ptl_src.x, ptl_src.m, ptl_dst.x)  # superposition rule
            forces.append(force)  # append the force value to step() every particle at once

        for (ptl_number, ptl) in enumerate(self.particles):
            force_current = forces[ptl_number]
            acc_current = force_current / ptl.m
            ptl.step(self.dt, acc_current)



        if self.save:
            """
            dim
            n_particle          // 여기까진 존재했었다?
            particles[0].m      // 예를 들어 2차원 시스템에서 두 개의 particle 이 있다? 
            particles[1].m
            particles[0].x[0]    
            particles[0].x[1]    
            particles[0].v[0]    
            particles[0].v[1]
            particles[1].x[0]
            particles[1].x[1]   
            particles[1].v[0]    
            particles[1].v[1]
            ...                 // 계속 이런식이다?
            """
            str_append = ''
            # str_append += (str(self.step_number) + '\n')
            if self.step_number == 0:
                for ptl in self.particles: # 질량 먼저 적는다
                    str_append += (str(ptl.m) + '\n')
                    print('fuck')

            for ptl in self.particles: # 그 다음 위치와 속도
                for i in range(len(ptl.x)):
                    str_append += (str(ptl.x[i]) + '\n')
                for j in range(len(ptl.v)):
                    str_append += (str(ptl.v[i]) + '\n')

            t_index_str = str(testcase_index).zfill(10) # 0으로 채워서 길이를 10으로 통일

            fd = open("../data/{0}/{1}.txt".format(self.name, t_index_str), 'a')  # appending 모드다?
            fd.write(str_append)
            fd.close()

        self.step_number += 1

    def check(self, min_distance):
        # check if the particles are too close or not
        for i, ptl1 in enumerate(self.particles):
            for ptl2 in self.particles[i+1:]:
                x1 = ptl1.x
                x2 = ptl2.x
                r_vec = x1 - x2
                r = np.sum(np.multiply(r_vec, r_vec)) ** 0.5

                if r < min_distance:
                    return False
        return True

    def plot(self):
        if self.dim != 2:
            print("Only 2 dimensional system can be plotted. Sorry!")
            return
        xs = [e.x[0] for e in self.particles]
        ys = [e.x[0] for e in self.particles]
        sizes = [100*e.m for e in self.particles]
        plt.scatter(xs, ys, s=sizes)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.show()

    def make_testcase(self, frame_number, testcase_number, min_distance):
        # make system's own directory
        dir = "../data/" + self.name
        if not os.path.exists(dir):
            os.makedirs(dir)
        for testcase_index in range(testcase_number):
            """
            <name.txt>
            dim
            n_particle
            """
            t_index_str = str(testcase_index).zfill(10)  # 0으로 채워서 길이를 10으로 통일
            fd = open("../data/{0}/{1}.txt".format(self.name, t_index_str), 'w')
            fd.write("{0}\n{1}\n".format(self.dim, self.n_particles))
            fd.close()

            for ptl in self.particles:
                x_new = np.array([random.uniform(-10, 10) for _ in range(self.dim)])
                v_new = np.array([random.uniform(-10, 10) for _ in range(self.dim)])
                m_new = random.uniform(0, 10)
                ptl.set_state(x_new, v_new, m_new)

            self.step_number = 0

            for _ in range(frame_number):
                if self.check(min_distance):
                    self.step(testcase_index)
                else:
                    print("testcase {0}: early termination!".format(t_index_str))
                    break

    def __repr__(self):
        return "{0}: {1} dimension system with {2} particles".format(
            self.name, self.dim, self.n_particles)