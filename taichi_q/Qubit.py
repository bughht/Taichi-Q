import numpy as np
import taichi as ti
import taichi.math as tm


class Qubits:
    """
    Qubits initalize (ti.Vec2.Field) as complex array
    """

    def __init__(self, num_qubits: int, qubit_state_init=0):
        self.num_qubits = num_qubits

        # Qubit States
        self.states = ti.Vector.field(2, ti.f64, [2] * self.num_qubits)

        # Assign Qubit States
        if isinstance(qubit_state_init, int):
            assert qubit_state_init in [
                0, 1
            ], "init state out of range, demand 0 or 1"
            self.states[tuple([qubit_state_init] * self.num_qubits)] = tm.vec2(
                1, 0)
        elif isinstance(qubit_state_init, (list, tuple, np.ndarray)):
            assert len(
                qubit_state_init
            ) == self.num_qubits, "init states shape mismatch with qubit num"
            self.states[tuple(qubit_state_init)] = tm.vec2(1, 0)

        self.measures = np.zeros(self.num_qubits, dtype=bool)
