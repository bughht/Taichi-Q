from taichi_q import Engine, Gate

if __name__ == "__main__":
    eng = Engine(
        num_qubits=2,
        state_init=0,
        device='gpu'
    )
    eng.State_Check()

    # Create Bell State
    eng.Ops(Gate.H(), [0])
    eng.Ops(Gate.X(), [1], [0])

    # Expect got |00> and |11>
    eng.State_Check(plot_state=True)

    # Cirucit
    eng.circuit_print()
    eng.circuit_visualize()
