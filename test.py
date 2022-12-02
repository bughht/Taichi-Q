import numpy as np
import taichi as ti

import taichi_q as tq
from taichi_q import Gate
from taichi_q.Engine import Engine

if __name__ == "__main__":
    eng = Engine(4, 1, ti.gpu, True)
    # eng.Ops(Gate.QFT(2), [0, 1])
    # eng.Ops(Gate.H(), [0])
    # eng.Ops(Gate.X(), [1], [0])
    eng.Ops(Gate.QFT(4), range(4))
    eng.Ops(Gate.iQFT(4), range(4))
