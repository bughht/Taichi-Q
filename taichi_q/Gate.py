from math import cos, pi, sin, sqrt

import numpy as np
import taichi as ti
import taichi.math as tm
from numpy import exp
from scipy.linalg import dft


@ti.data_oriented
class GateBase:
    """
    Base class of quantum gate
    """

    def __init__(self, mat: np.array):
        self.q_num = int(np.log2(mat.shape[0]))
        self.matrix = ti.Vector.field(2, ti.f64, mat.shape)
        self.ndarray2field(mat.real, mat.imag)

    # @property
    # def matrix(self):
    #     return self.matrix

    @ti.kernel
    def ndarray2field(
            self,
            mat_r: ti.types.ndarray(),
            mat_i: ti.types.ndarray()):
        for i, j in self.matrix:
            self.matrix[i, j] = tm.vec2(mat_r[i, j], mat_i[i, j])


class H(GateBase):
    def __init__(self):
        super().__init__(H_)


class X(GateBase):
    def __init__(self):
        super().__init__(X_)


class Y(GateBase):
    def __init__(self):
        super().__init__(Y_)


class Z(GateBase):
    def __init__(self):
        super().__init__(Z_)


class S(GateBase):
    def __init__(self):
        super().__init__(S_)


class T(GateBase):
    def __init__(self):
        super().__init__(T_)


class S(GateBase):
    def __init__(self):
        super().__init__(S_)


class T(GateBase):
    def __init__(self):
        super().__init__(T_)


class Rx(GateBase):
    def __init__(self, theta):
        super().__init__(Rx_(theta))


class Ry(GateBase):
    def __init__(self, theta):
        super().__init__(Ry_(theta))


class Rz(GateBase):
    def __init__(self, theta):
        super().__init__(Rz_(theta))


class QFT(GateBase):
    def __init__(self, n):
        super().__init__(QFT_(n))


c128 = np.complex128
H_ = np.array([[1, 1],
               [1, -1]], dtype=c128)/sqrt(2)
X_ = np.array([[0, 1],
              [1, 0]], dtype=c128)
Y_ = np.array([[0, 0-1j],
               [0+1j, 0]], dtype=c128)
Z_ = np.array([[1, 0],
               [1, -1]], dtype=c128)
S_ = np.array([[1, 0],
               [0, 0+1j]], dtype=c128)
T_ = np.array([[1, 0],
               [0, exp(0+1j*(pi/4))]], dtype=c128)


def Rx_(theta): return np.array(
    [[cos(theta/2), -1j*sin(theta/2)],
     [-1j*sin(theta/2), cos(theta/2)]], dtype=c128)


def Ry_(theta): return np.array(
    [[cos(theta/2), -sin(theta/2)],
     [sin(theta/2), cos(theta/2)]], dtype=c128)


def Rz_(theta): return np.array(
    [[exp(-1j*theta/2), 0],
     [0, exp(1j*theta/2)]], dtype=c128)


def QFT_(n): return np.array(dft(2**n, 'sqrtn'), dtype=c128)
