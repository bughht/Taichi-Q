import numpy as np

from taichi_q import Engine, Gate

if __name__ == "__main__":
    eng = Engine(
        num_qubits=3,
        state_init=[[-1/np.sqrt(5), 2j/np.sqrt(5)], [1, 0], [1, 0]],
        device='cpu')
    eng.State_Check()
    eng.Ops(Gate.H(), [1])
    eng.circuit_print()
    eng.State_Check()
    eng.Ops(Gate.X(), [2], [1])
    eng.circuit_print()
    eng.State_Check()
    eng.Ops(Gate.X(), [1], [0])
    eng.circuit_print()
    eng.State_Check()
    eng.Ops(Gate.H(), [0])
    eng.circuit_print()
    eng.qubits.cheat()
    eng.Measure(0)
    eng.State_Check()
    eng.circuit_print()
    eng.Measure(1)
    eng.State_Check()
    eng.circuit_print()
    eng.Ops(Gate.X(), [2], [1])
    eng.State_Check()
    eng.circuit_print()
    eng.Ops(Gate.Z(), [2], [0])
    eng.State_Check(plot_state=True)
    eng.circuit_print()
    eng.circuit_visualize()
