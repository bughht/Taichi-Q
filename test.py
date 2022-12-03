import numpy as np
import taichi as ti

import taichi_q as tq
from taichi_q import Gate
from taichi_q.Engine import Engine

if __name__ == "__main__":
    eng = Engine(
        4,
        [[-1/np.sqrt(2), 1j/np.sqrt(2)],
         [1, 0], [1, 0], [1, 0]],
        ti.cpu,
        True)
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
    eng.Ops(Gate.swap(), [0, 1], [2])
    eng.qubits.cheat()
    print(eng.gate_state[:, :9])
    eng.Measure(0)
    eng.State_Check(True, False)
    eng.Measure(1)
    eng.qubits.cheat()
    print(eng.gate_state[:, :9])
    eng.Ops(Gate.X(), [2], [1])
    eng.qubits.cheat()
    print(eng.gate_state[:, :9])
    eng.Ops(Gate.Z(), [2], [0])
    print(eng.gate_state[:, :9])
    eng.circuit_visualize()

    # eng.Measure(2)
    # eng.qubits.cheat()
    # eng.Ops(Gate.QFT(2), range(2))
    # eng.qubits.cheat()
    # eng.Ops(Gate.iQFT(2), range(2))
    # eng.qubits.cheat()
    # eng.circuit_visualize()
