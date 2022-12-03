import numpy as np

from taichi_q import Engine, Gate

if __name__ == "__main__":
    eng = Engine(num_qubits=3, state_init=0, device='cpu')

    # Uniform
    for i in range(3):
        eng.Ops(Gate.H(), [i])
    eng.State_Check(plot_state=True)

    # Phase Oracle 101 and 110
    eng.Ops(Gate.Z(), [0], [2])
    eng.Ops(Gate.Z(), [0], [1])
    eng.State_Check(plot_state=True)

    # Amplification
    for i in range(3):
        eng.Ops(Gate.H(), [i])
        eng.Ops(Gate.X(), [i])
    eng.Ops(Gate.Z(), [0], [1, 2])
    for i in range(3):
        eng.Ops(Gate.X(), [i])
        eng.Ops(Gate.H(), [i])
    eng.State_Check(plot_state=True)

    # Measure
    result = np.zeros(3)
    for i in range(3):
        result[i] = eng.Measure(i)
    print(result)

    # Circuit
    eng.circuit_print()
    eng.circuit_visualize()
