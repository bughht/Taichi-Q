import numpy as np
import taichi as ti
import taichi.math as tm

from taichi_q import Gate, Qubits


class Engine:
    """
    Quantum Computation Simulator Engine
    """

    def __init__(self,
                 num_qubits,
                 state_init=0,
                 device=ti.cuda,
                 debug=False):
        """
            Initialize Engine, define qubit num and device

        Args:
            num_qubits (int): Qubit Num
            state_init (int/Arraylike, optional): Initialize state of qubits. Defaults to 0.
            device (_type_, optional): Device for simulator to run, support ti.cpu / gpu (cuda only). Defaults to ti.cpu.
            debug (bool, optional): Debugmode for taichi init. Defaults to False.
        """
        self.num_qubits = num_qubits
        self.state_init = state_init
        self.device = device
        self.debug = debug

        ti.init(arch=device, default_fp=ti.f64,
                dynamic_index=True, debug=debug)
        self.qubits_init()
        # self.bit_init()

    def qubits_init(self):
        self.qubits = Qubits(self.num_qubits, self.state_init, self.device)

    def Ops(self, ops, target, control=[]):
        target = np.asarray(target, dtype=int)
        control = np.asarray(control, dtype=int)
        assert ops.q_num == len(target), "Gate size mismatch with Qubit num"

        self.qubits.Ops(ops,
                        target, control)

    def Measure(self, target):
        target = np.asarray(target, dtype=int)
        assert all(target >= 0) and all(
            target < self.num_qubits), 'Target Qubit not in range'
