from typing import List

from Algorithm import QuantumAlgorithm
import Gate
import Circuit
from .util import convert_int_to_list, convert_list_to_int
import numpy as np
import copy

'''
Simon's algorithm
The input is a function f and a number s,
such that f(x1)==f(x2) iff x1+x2=s of x1==x2
Our goal is to find such s
'''


class Simon(QuantumAlgorithm):

    def __init__(self, num_qubits: int) -> None:
        if (num_qubits % 2) != 0:
            raise ValueError("The number of qubits must be even in Simon problem")
        super().__init__(num_qubits)
        self.circuit = Circuit.NumpyCircuit(num_qubits)
        self.num_qubits = num_qubits
        self.uf = []
        self._s = 0

    def construct_circuit(self) -> None:
        '''
        For convenience, we use Allhadamard gats instead of half of the hadamard. This does not
        affect the result since we only measure the above half
        '''
        self.circuit.add_gate(Gate.AllHadamard(self.num_qubits), list(range(0, self.num_qubits)))
        self.compile_uf()
        self.circuit.add_gate(Gate.AllHadamard(self.num_qubits), list(range(0, self.num_qubits)))

    def set_input(self, uf: List) -> None:
        func_dim = int(self.num_qubits / 2)
        if len(uf) != (1 << func_dim):
            raise ValueError("The input dimension does not match!")
        self._uf = uf

    def compile_uf(self) -> None:
        func_dim = int(self.num_qubits / 2)
        for i in range(len(self._uf)):
            value = self._uf[i]
            value_bit = convert_int_to_list(func_dim, value)
            input_bit = convert_int_to_list(func_dim, i)
            for j in range(func_dim):
                '''
                Only add controlX gate when the function value is 1 
                '''
                if value_bit[j] == 1:
                    if self.num_qubits == 2:
                        self.circuit.add_gate(Gate.CNOT(), [0, 1])
                    else:
                        self.circuit.add_gate(
                            Gate.MultiControlX(func_dim + 1, input_bit),
                            [list(range(0, func_dim)), func_dim + j])
        return

    def compute_result(self) -> None:
        enough = False
        func_dim = int(self.num_qubits / 2)
        measure_indices = list(range(0, func_dim))
        y_result = []
        pre_rank = 0
        while not enough:
            self.circuit.compute()
            result = self.circuit.measure(measure_indices)
            matrix = copy.copy(y_result)
            matrix.append(result)
            rank = np.linalg.matrix_rank(np.array(matrix))
            if rank > pre_rank:
                y_result.append(result)
                pre_rank = rank
                if rank == func_dim - 1:
                    enough = True

        a = np.array(y_result)
        b = np.array([0] * func_dim)
        s = np.linalg.solve(a, b)
        print(f"The solution for the given function is s={s}")
        return
