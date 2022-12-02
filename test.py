import numpy as np
import taichi as ti

import taichi_q as tq
from taichi_q import Gate
from taichi_q.Engine import Engine

if __name__ == "__main__":
    eng = Engine(3, 0, ti.cpu, True)
    eng.Ops(Gate.X(), [0])
    eng.qubits.cheat()
    eng.Ops(Gate.H(), [0])
    eng.qubits.cheat()
    eng.Ops(Gate.X(), [1], [0])
    eng.qubits.cheat()
    eng.Ops(Gate.H(), [0])
    eng.qubits.cheat()
    eng.Ops(Gate.X(), [2], [1])
    eng.qubits.cheat()
    eng.Ops(Gate.Z(), [2], [0])
    eng.qubits.cheat()
    eng.Ops(Gate.QFT(3), range(3))
    eng.qubits.cheat()
    # eng.Ops(Gate.iQFT(4), range(4), 4)
    # eng.qubits.cheat()
