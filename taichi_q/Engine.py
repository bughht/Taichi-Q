import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
import taichi.math as tm

from taichi_q import Qubits

# """

# Args:
#     num_qubits (int): Qubit Num
#     state_init (int/Arraylike, optional): Initialize state of qubits. Defaults to 0.
#     device (_type_, optional): Device for simulator to run, support ti.cpu / gpu. Defaults to ti.cpu.
#     debug (bool, optional): Debugmode for taichi init. Defaults to False.
# """


@ti.data_oriented
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
        for tgt in target:
            self.gate_state[tgt, self.gate_num] = ops_name
        for ctl in control:
            self.gate_state[ctl, self.gate_num] = '■'
        self.gate_num += 1

    def qubit_state_demonstrate(self):
        """
        Demonstrate qubit state init
        """
        self.state_dem = np.zeros((self.num_qubits, 2), dtype=np.float32)
        if isinstance(self.state_init, int):
            assert self.state_init in [
                0, 1
            ], "init state out of range, demand 0 or 1"
            self.state_dem[:, self.state_init] = 1
        elif isinstance(self.state_init, (list, tuple, np.ndarray)):

            assert len(
                self.state_init
            ) == self.num_qubits, "init states shape mismatch with qubit num"
            if isinstance(self.state_init[0], int):
                self.state_dem[:, self.state_init] = 1
            else:
                self.state_init = np.asarray(self.state_init)
                assert self.state_init.shape[0] == self.num_qubits and self.state_init.shape[
                    1] == 2, "init states shape not available for qubits"
                assert np.allclose(np.square(np.abs(self.state_init)).sum(
                    axis=1), np.ones(self.num_qubits)), "init states not available for qubits (sum p = not equal 1), please check the states again"
                self.state_dem = np.square(np.abs(self.state_init))

    def circuit_visualize(self):
        """
        Visualize the Quantum Circuit
        """
        self.qubit_state_demonstrate()
        self.pixels = ti.Vector.field(3, dtype=ti.f64, shape=(
            (self.gate_num+3)*100, (self.num_qubits+1)*100))
        gui = ti.GUI("Taichi-Q", res=self.pixels.shape)
        t = 0.0
        while gui.running:
            t += 0.1
            self.background(t)
            gui.set_image(self.pixels)
            for qubit_line in range(self.num_qubits):
                gui.line(
                    [2./(self.gate_num+3), (qubit_line+1)/(self.num_qubits+1.)],
                    [(self.gate_num+2)/(self.gate_num+3),
                     (qubit_line+1)/(self.num_qubits+1.)],
                    radius=3,
                    color=0x000000)
                gui.text(
                    pos=[30./self.pixels.shape[0], (qubit_line+1)/(self.num_qubits+1.) +
                         15/self.pixels.shape[1]],
                    content="   Qubit{}".format(self.num_qubits-1-qubit_line),
                    color=0x000000,
                    font_size=30)
                gui.text(
                    pos=[20./self.pixels.shape[0], (qubit_line+1)/(self.num_qubits+1.) -
                         15/self.pixels.shape[1]],
                    content="{:.1f}|0>+{:.1f}|1>".format(
                        self.state_dem[self.num_qubits-1-qubit_line, 0], self.state_dem[self.num_qubits-1-qubit_line, 1]),
                    color=0x000000,
                    font_size=30)
            for gate in range(self.gate_num):
                gate_pos = np.where(self.gate_state[:, gate] != ' ')[0]
                if len(gate_pos) > 1:
                    gui.line(
                        [(gate+2.5)/(self.gate_num+3)-5/self.pixels.shape[0],
                         (self.num_qubits-1-gate_pos.min()+1)/(self.num_qubits+1)+0/self.pixels.shape[1]],
                        [(gate+2.5)/(self.gate_num+3)-5/self.pixels.shape[0],
                         (self.num_qubits-1-gate_pos.max()+1)/(self.num_qubits+1)+0/self.pixels.shape[1]],
                        radius=3,
                        color=0x000000
                    )
                for gate_idx in gate_pos:
                    pos = self.num_qubits-1-gate_idx
                    if self.gate_state[gate_idx, gate] == '■':
                        gui.circle(
                            pos=[(gate+2.5)/(self.gate_num+3)-5/self.pixels.shape[0],
                                 (pos+1)/(self.num_qubits+1)+0/self.pixels.shape[1]],
                            color=0x000000,
                            radius=10
                        )
                    elif self.gate_state[gate_idx, gate] == 's':
                        gui.line(
                            [(gate+2.5)/(self.gate_num+3)-30/self.pixels.shape[0],
                             (pos+1)/(self.num_qubits+1)+25/self.pixels.shape[1]],
                            [(gate+2.5)/(self.gate_num+3)+20/self.pixels.shape[0],
                             (pos+1)/(self.num_qubits+1)-25/self.pixels.shape[1]],
                            radius=3,
                            color=0x000000
                        )
                        gui.line(
                            [(gate+2.5)/(self.gate_num+3)+20/self.pixels.shape[0],
                             (pos+1)/(self.num_qubits+1)+25/self.pixels.shape[1]],
                            [(gate+2.5)/(self.gate_num+3)-30/self.pixels.shape[0],
                             (pos+1)/(self.num_qubits+1)-25/self.pixels.shape[1]],
                            radius=3,
                            color=0x000000
                        )
                    elif self.gate_state[gate_idx, gate] == 'M':
                        gui.line(
                            [(gate+2.5)/(self.gate_num+3),
                                (pos+1)/(self.num_qubits+1.)],
                            [(self.gate_num+2)/(self.gate_num+3), (pos+1)/(self.num_qubits+1.)], radius=6, color=0x000000)
                        self.rect_colored(
                            gui,
                            topleft=[(gate+2.5)/(self.gate_num+3)-30/self.pixels.shape[0],
                                     (pos+1)/(self.num_qubits+1)+25/self.pixels.shape[1]],
                            bottomright=[(gate+2.5)/(self.gate_num+3)+20/self.pixels.shape[0],
                                         (pos+1)/(self.num_qubits+1)-25/self.pixels.shape[1]],
                            radius=6,
                            linecolor=0x000000,
                            color=0xFFFFFF)
                        gui.text(
                            pos=[(gate+2.5)/(self.gate_num+3)-20/self.pixels.shape[0],
                                 (pos+1)/(self.num_qubits+1)+20/self.pixels.shape[1]],
                            content=self.gate_state[gate_idx, gate],
                            color=0x000000,
                            font_size=40
                        )
                    else:
                        self.rect_colored(
                            gui,
                            topleft=[(gate+2.5)/(self.gate_num+3)-30/self.pixels.shape[0],
                                     (pos+1)/(self.num_qubits+1)+25/self.pixels.shape[1]],
                            bottomright=[(gate+2.5)/(self.gate_num+3)+20/self.pixels.shape[0],
                                         (pos+1)/(self.num_qubits+1)-25/self.pixels.shape[1]],
                            radius=3,
                            linecolor=0x000000,
                            color=0xFFFFFF)
                        gui.text(
                            pos=[(gate+2.5)/(self.gate_num+3)-20/self.pixels.shape[0],
                                 (pos+1)/(self.num_qubits+1)+20/self.pixels.shape[1]],
                            content=self.gate_state[gate_idx, gate],
                            color=0x000000,
                            font_size=40
                        )
            gui.show()

    def rect_colored(self, gui, topleft, bottomright, radius, linecolor, color):
        topright = [bottomright[0], topleft[1]]
        bottomleft = [topleft[0], bottomright[1]]
        gui.triangles(
            a=np.array([topleft, topleft]),
            b=np.array([topright, bottomleft]),
            c=np.array([bottomright, bottomright]),
            color=color
        )
        gui.rect(
            topleft=topleft,
            bottomright=bottomright,
            radius=radius,
            color=linecolor
        )

    @ ti.kernel
    def background(self, t: ti.f64):
        self.pixels.fill(tm.vec3(
            0.92+0.08*tm.sin(t),
            0.92+0.08*tm.sin(t+tm.pi*2/3),
            0.92+0.08*tm.sin(t+tm.pi*4/3)))

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
        return result

    def State_Check(self, print_state=True, plot_state=False) -> dict:
        states = self.qubits.cheat(print_state)
        if plot_state:
            plt.figure(figsize=(10, 5))
            plt.bar(states['Q'], states['P'], color='maroon')
            plt.xticks(rotation=25)
            plt.ylabel('P')
            # plt.ylim(0, 1)
            plt.show()
