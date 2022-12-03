import numpy as np
import taichi as ti

import taichi_q as tq
from taichi_q import Gate
from taichi_q.Engine import Engine

if __name__ == "__main__":
    eng = Engine(3, [[-1/np.sqrt(2), 1j/np.sqrt(2)],
                 [1, 0], [1, 0]], ti.cpu, True)
    eng.qubits.cheat()
    eng.Ops(Gate.H(), [1])
    print(eng.gate_state[:, :8])
    eng.qubits.cheat()
    eng.Ops(Gate.X(), [2], [1])
    print(eng.gate_state[:, :8])
    eng.qubits.cheat()
    eng.Ops(Gate.X(), [1], [0])
    print(eng.gate_state[:, :8])
    eng.qubits.cheat()
    eng.Ops(Gate.H(), [0])
    print(eng.gate_state[:, :8])
    eng.qubits.cheat()
    eng.Measure(0)
    eng.qubits.cheat()
    eng.Measure(1)
    eng.qubits.cheat()
    print(eng.gate_state[:, :8])
    eng.Ops(Gate.X(), [2], [1])
    eng.qubits.cheat()
    eng.Ops(Gate.Z(), [2], [0])
    eng.qubits.cheat()
    print(eng.gate_state[:, :8])

    # eng.Measure(2)
    # eng.qubits.cheat()
    # eng.Ops(Gate.iQFT(4), range(4), 4)
    # eng.qubits.cheat()
