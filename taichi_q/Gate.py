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

    def __init__(self, name: str, mat: np.array):
        """
        Initialize GateBase with gate name and its matrix

        Args:
            name (str): gate name
            mat (np.array): gate matrix (Unitary unless measure gate)
        """
        if name != 'Measure':
            assert np.allclose(np.eye(
                mat.shape[0]), mat@np.conj(mat).T), "Please make sure the gate matrix is unitary"
        self.name = name
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
        """
        Convert complex128 ndarray into f64 tm.vec2.fields

        Args:
            mat_r (ti.types.ndarray): Real part for complex matrix
            mat_i (ti.types.ndarray): Imaginary part for complex matrix
        """
        for i, j in self.matrix:
            self.matrix[i, j] = tm.vec2(mat_r[i, j], mat_i[i, j])


class H(GateBase):
    """
    H Gate
    """

    def __init__(self):
        super().__init__('H', H_)


class X(GateBase):
    """
    X Gate
    """

    def __init__(self):
        super().__init__('X', X_)


class Y(GateBase):
    """
    Y Gate
    """

    def __init__(self):
        super().__init__('Y', Y_)


class Z(GateBase):
    """
    Z Gate
    """

    def __init__(self):
        super().__init__('Z', Z_)


class S(GateBase):
    """
    S Gate
    """

    def __init__(self):
        super().__init__('S', S_)


class T(GateBase):
    """
    T Gate
    """

    def __init__(self):
        super().__init__('T', T_)


class swap(GateBase):
    """
    swap Gate
    """

    def __init__(self):
        super().__init__('swap', swap_)


class U(GateBase):
    """
    U Gate
    """

    def __init__(self, theta, phi, lamb):
        super().__init__('U', U_(theta, phi, lamb))


class Rx(GateBase):
    """
    Rx(theta) Gate
    """

    def __init__(self, theta):
        super().__init__('Rx(theta)', Rx_(theta))


class Ry(GateBase):
    """
    Ry(theta) Gate
    """

    def __init__(self, theta):
        super().__init__('Ry(theta)', Ry_(theta))


class Rz(GateBase):
    """
    Rz(theta) Gate
    """

    def __init__(self, theta):
        super().__init__('Rz(theta)', Rz_(theta))


class QFT(GateBase):
    """
    QFT Gate
    """

    def __init__(self, n):
        super().__init__('QFT', QFT_(n))


class iQFT(GateBase):
    """
    iQFT Gate
    """

    def __init__(self, n):
        super().__init__('iQFT', iQFT_(n))


class Measure(GateBase):
    """
    Measure
    """

    def __init__(self, p0, p1):
        sample = np.random.random()
        if sample < p0:
            self.result = 0
            super().__init__('Measure', Measure_0(p0, p1))
        else:
            self.result = 1
            super().__init__('Measure', Measure_1(p0, p1))


c128 = np.complex128
H_ = np.array([[1, 1],
               [1, -1]], dtype=c128)/sqrt(2)
X_ = np.array([[0, 1],
              [1, 0]], dtype=c128)
Y_ = np.array([[0, 0-1j],
               [0+1j, 0]], dtype=c128)
Z_ = np.array([[1, 0],
               [0, -1]], dtype=c128)
S_ = np.array([[1, 0],
               [0, 0+1j]], dtype=c128)
T_ = np.array([[1, 0],
               [0, exp(0+1j*(pi/4))]], dtype=c128)
swap_ = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]], dtype=c128)


def U_(theta, phi, lamb): return np.array([[np.cos(theta/2), -np.exp(1j*lamb)*np.sin(theta/2)],
                                          [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lamb))*np.cos(theta/2)]])


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


def iQFT_(n): return np.conj(QFT_(n))


def Measure_0(p0, p1): return np.array([[1/np.sqrt(p0), 0],
                                        [0, 0]], dtype=c128)


def Measure_1(p0, p1): return np.array([[0, 0],
                                        [0, 1/np.sqrt(p1)]], dtype=c128)
