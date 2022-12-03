import numpy as np

from taichi_q import Engine, Gate

if __name__ == "__main__":
    eng = Engine(
        num_qubits=3,
        state_init=[[-1/np.sqrt(5), 2j/np.sqrt(5)], [1, 0], [1, 0]],
        device='cpu')

    eng.State_Check(plot_state=True)
    # Step 1: A Third Party [Telamon] created an entangled qubit pair. One to Alice, another to Bob
    eng.Ops(Gate.H(), [1])
    eng.Ops(Gate.X(), [2], [1])
    eng.State_Check()

    # Step 2: Alice applies CNOT to q1 controlled by q0 (qubit send to Bob) and H to q0
    eng.Ops(Gate.X(), [1], [0])
    eng.Ops(Gate.H(), [0])
    eng.State_Check()

    # Step 3: Alice measure both q0 and q1 and send measure results to Bob
    eng.Measure(0)
    eng.Measure(1)
    eng.State_Check()

    # Step 4: Bob got q2 and measure result from Alice, then applied Gate.
    eng.Ops(Gate.X(), [2], [1])
    eng.Ops(Gate.Z(), [2], [0])

    # q0 states successfully teleported to q2
    eng.State_Check(plot_state=True)

    eng.circuit_print()
    eng.circuit_visualize()
