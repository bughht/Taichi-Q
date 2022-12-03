import numpy as np
import taichi as ti
import taichi.math as tm

from taichi_q import Gate, Qubits

# """

# Args:
#     num_qubits (int): Qubit Num
#     state_init (int/Arraylike, optional): Initialize state of qubits. Defaults to 0.
#     device (_type_, optional): Device for simulator to run, support ti.cpu / gpu. Defaults to ti.cpu.
#     debug (bool, optional): Debugmode for taichi init. Defaults to False.
# """


class Engine:
    """
    Quantum Computation Simulator Engine
    """

    def __init__(self,
                 num_qubits,
                 state_init=0,
                 device=ti.cpu,
                 debug=False):
        """
            Initialize Engine, define qubit num and device

        Args:
            num_qubits (int): Qubit Num
            state_init (int/Arraylike, optional): Initialize state of qubits. Defaults to 0. 
            device (ti.misc, optional): Device for simulator to run, support ti.cpu / gpu. Defaults to ti.cpu.
            debug (bool, optional): Debugmode for taichi init. Defaults to False.
        """

        self.num_qubits = num_qubits
        self.state_init = state_init
        self.device = device
        self.debug = debug

        ti.init(arch=device, default_fp=ti.f64,
                dynamic_index=True, debug=debug)
        self.qubits_init()
        self.max_gate = 1024
        self.gate_states_init(self.max_gate)

    def qubits_init(self):
        """
        Initialize Qubit states
        """
        self.qubits = Qubits(self.num_qubits, self.state_init, self.device)

    def gate_states_init(self, max_gate=1024):
        """
        Initialize gate states (for circuit visualization)

        Args:
            max_gate (int, optional): max gate num. Defaults to 1024.
        """
        self.gate_state = np.empty((self.num_qubits, max_gate), dtype=str)
        self.gate_state.fill(' ')

        self.gate_num = 0

    def gate_states_append(self, ops_name, target, control=[]):
        """
        Update Gate Status

        Args:
            ops_name (str): operator label
            target (list/arraylike): target qubit
            control (list/arraylike, optional): controlled qubit. Defaults to [].
        """
        # offset = []
        # for tgt in target:
        #     history = np.where(self.gate_state[tgt] != ' ')[0]
        #     print(history.shape)
        #     if history.shape[0] > 0:
        #         offset.append(self.gate_num-history.max()-1)
        # for ctl in control:
        #     history = np.where(self.gate_state[ctl] != ' ')[0]
        #     if history.shape[0] > 0:
        #         offset.append(self.gate_num-history.max()-1)
        # if len(offset) == 0:
        #     bk_offset = 0
        # else:
        #     bk_offset = min(offset)
        for tgt in target:
            self.gate_state[tgt, self.gate_num] = ops_name
        for ctl in control:
            self.gate_state[ctl, self.gate_num] = '■'
        self.gate_num += 1

    def Ops(self, ops, target, control=[]):
        """
        Operate Quantum Gate to specific qubits
        Args:
            ops (taichi_q.Gate.GateBase): quantum gate 
            target (list): target qubits for quantum gate operations
            control (list, optional): qubits for controlled gate
        """
        target = np.asarray(target, dtype=int)
        control = np.asarray(control, dtype=int)
        assert ops.q_num == len(target), "Gate size mismatch with Qubit num"
        assert all(self.qubits.measured[target]
                   == 0), 'Target Qubit already collapsed'
        self.gate_states_append(ops.name, target, control)

        self.qubits.Ops(ops,
                        target, control)

    def Measure(self, target: int) -> int:
        """
        Measure the state of target qubit, project a single qubit into |0> or |1> state

        Args:
            target (int): target qubit to measure

        Returns:
            int: measurement result
        """
        assert target >= 0 and target < self.num_qubits, 'Target Qubit not in range'
        assert self.qubits.measured[target] == 0, 'Target Qubit already collapsed'
        self.gate_states_append('M', [target])
        result = self.qubits.Measure(target)
        print(result)
