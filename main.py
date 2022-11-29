import taichi as ti

import taichi_q as tq
from taichi_q.Engine import Engine
import numpy as np

if __name__ == "__main__":
    eng = Engine(2, np.array([1, 0]), ti.cpu, True)
    print(eng.qubits)
