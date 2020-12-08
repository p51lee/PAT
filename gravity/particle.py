import numpy as np


class IdenticalParticle: # particle with size 0
    def __init__(self,
                 m: float = 1.,
                 x_init: np.ndarray = np.array([0, 0]),
                 v_init: np.ndarray = np.array([0, 0])):
        """(particle number, mass, position vector, velocity vector)"""
        self.m = m
        self.x = np.array(x_init)
        self.v = np.array(v_init)
        self.dim = len(x_init)

    # def set_number(self, n: int):
    #     self.ptl_number = n;
    #     return

    def set_state(self, x_in: np.ndarray, v_in: np.ndarray, m_in: float):
        if len(x_in) != len(self.x) or len(v_in) != len(self.v):
            return False
        self.x = x_in
        self.v = v_in
        self.m = m_in
        return True

    def step(self, dt: float, a: np.ndarray):
        self.x += self.v * dt
        self.v += a * dt
        return True

    def __repr__(self):
        return "{0} kg particle with position {1} and velocity {2}".format(
            self.m, self.x, self.v)

# p = IdenticalParticle()
# print(p)
