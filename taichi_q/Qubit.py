import numpy as np
import taichi as ti
import taichi.math as tm


@ti.data_oriented
class Qubits:
    """
    Qubits initalize (ti.Vec2.Field) as complex array
    TODO: Replace Dense Field with Sparse Field
    """

    def __init__(self, num_qubits: int, qubit_state_init=0, device=ti.cpu):
        self.num_qubits = num_qubits

        # Qubit States
        self.states = ti.Vector.field(2, ti.f64, [2] * self.num_qubits)
        self.device = device

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

        self.measures = np.zeros(self.num_qubits, dtype=np.int8)

        self.external_init()

    def external_init(self):
        self.zeros_vec = np.zeros(self.num_qubits, dtype=np.int8)
        self.zeros_mat = np.zeros(
            (self.num_qubits, self.num_qubits), dtype=np.int8)

    @ti.kernel
    def mat_gen(self, mat: ti.template(), mat_ops: ti.template(), ctl_num: ti.int32):
        for i in range(ctl_num):
            mat[i, i] = tm.vec2([1, 0])
        for i, j in mat_ops:
            mat[ctl_num+i, ctl_num+j] = mat_ops[i, j]

    def Ops(self, ops, target, control):
        tgt = np.hstack([control, target])
        mat = ti.Vector.field(2, ti.f64, [2**len(tgt), 2**len(tgt)])
        self.mat_gen(mat, ops.matrix, mat.shape[0]-ops.matrix.shape[0])
        self.len_in = len(tgt)
        self.len_ex = self.num_qubits-self.len_in
        print('in', self.len_in, 'ex', self.len_ex)

        self.qubit_ops = ti.Vector.field(
            self.num_qubits, int, 2**self.len_in)
        self.state_ops = ti.Vector.field(
            2, ti.f64, 2**self.len_in)

        if len(tgt) == self.num_qubits:
            self.Ops_kernel_full(mat, tgt)
        else:
            self.Ops_kernel_part(mat, tgt)

    @ti.kernel
    def Ops_kernel_part(
            self,
            mat: ti.template(),
            target: ti.types.ndarray()):
        trans_Mat = ti.Matrix(self.zeros_mat)
        # ti.loop_config(serialize=True)
        for t in target:
            trans_Mat[target[t], self.len_ex+t] = 1
        ex_offset = 0
        ti.loop_config(serialize=True)
        for i in range(self.len_ex):
            flag = -1
            while flag != 0:
                flag = 0
                for j in range(self.num_qubits):
                    flag += trans_Mat[i+ex_offset, j]
                if flag != 0:
                    ex_offset += 1
            trans_Mat[i+ex_offset, i] = 1

        ti.loop_config(serialize=True)
        for I_ex in ti.grouped(ti.ndrange(*[2]*self.len_ex)):
            # ti.loop_config(serialize=True)
            for I_in in ti.grouped(ti.ndrange(*[2]*self.len_in)):
                idx = ti.Vector(self.zeros_vec)

                for i in ti.ndrange(self.len_ex):
                    idx[i] = I_ex[i]
                for i in ti.ndrange(self.len_in):
                    idx[self.len_ex+i] = I_in[i]
                # if self.device == ti.cpu:
                idx_i = 0
                for i in ti.static(range(self.len_in)):
                    idx_i += idx[i+self.len_ex]*2**(self.len_in-1-i)

                print(ti.global_thread_idx(), trans_Mat, idx)
                idx = trans_Mat@idx

                print(ti.global_thread_idx(), idx_i, idx, self.states[idx])
                self.qubit_ops[idx_i] = idx
                self.state_ops[idx_i] = self.states[idx]

            # self.cmat(mat)
            nozero_flag = 0
            for i in range(2**self.len_in):
                print(ti.global_thread_idx(),
                      self.qubit_ops[i], self.state_ops[i])
                if any(self.state_ops[i] != 0):
                    nozero_flag = 1
                    # break
            if nozero_flag != 0:
                for i in range(2**self.len_in):
                    sum_ = tm.vec2(0, 0)
                    for j in range(2**self.len_in):
                        sum_ += tm.cmul(mat[i, j], self.state_ops[j])
                    self.states[self.qubit_ops[i]] = sum_

    @ti.kernel
    def Ops_kernel_full(
            self,
            mat: ti.template(),
            target: ti.types.ndarray()):
        trans_Mat = ti.Matrix(self.zeros_mat)

        for t in target:
            trans_Mat[target[t], t] = 1

        # ti.loop_config(serialize=True)
        for I_in in ti.grouped(self.states):
            idx = I_in
            idx_i = 0
            for i in range(self.len_in):
                idx_i += idx[i]*2**(self.len_in-1-i)

            idx = trans_Mat@idx
            # print(idx_i, idx, self.states[idx])
            self.qubit_ops[idx_i] = idx
            self.state_ops[idx_i] = self.states[idx]
            idx_i += 1

        self.cmat(mat)

    @ti.kernel
    def cheat(self):
        sum = 0.0
        for I in ti.grouped(self.states):
            print(I, self.states[I], self.cabs(self.states[I]))
            sum += self.cabs(self.states[I])
        print('SUM:', sum)

    @staticmethod
    @ti.func
    def cabs(c: tm.vec2) -> ti.f32:
        return tm.cmul(c, tm.cconj(c))[0]

    @ti.func
    def cmat(self, mat):
        nozero_flag = 0
        for i in range(2**self.len_in):
            if any(self.state_ops[i] != 0):
                nozero_flag = 1
                # break
        if nozero_flag != 0:
            for i in range(2**self.len_in):
                sum_ = tm.vec2(0, 0)
                for j in range(2**self.len_in):
                    sum_ += tm.cmul(mat[i, j], self.state_ops[j])
                self.states[self.qubit_ops[i]] = sum_
