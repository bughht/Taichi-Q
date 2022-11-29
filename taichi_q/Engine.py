import numpy as np
import taichi as ti
import taichi.math as tm

from taichi_q import Qubits


class Engine:
    """
    Quantum Computation Simulator Engine
    """

    def __init__(self,
                 num_qubits,
                 state_init=0,
                 device=ti.cuda,
                 debug=False) -> None:
        """
            Initialize Engine, define qubit num and device

        Args:
            num_qubits (int): Qubit Num
            state_init (int/Arraylike, optional): Initialize state of qubits. Defaults to 0.
            device (_type_, optional): Device for simulator to run, support ti.cpu (x64, x86_64, arm64, cc, wasm) / gpu (cuda, metal, vulkan, opengl, dx11, dx12). Defaults to ti.cpu.
            debug (bool, optional): Debugmode for taichi init. Defaults to False.
        """

        self.num_qubits = num_qubits
        self.state_init = state_init
        self.device = device
        self.debug = debug

        ti.init(arch=device, debug=debug)
        self.qubits_init()

    def qubits_init(self):
        self.qubits = Qubits(self.num_qubits, self.state_init)
