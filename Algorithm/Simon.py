from typing import List

from Algorithm import QuantumAlgorithm
import Gate
import Circuit
from .util import convert_int_to_list, convert_list_to_int
import numpy as np
import copy
import qiskit
import functools
from qiskit_aer import AerSimulator
from itertools import product
'''
Simon's algorithm
The input is a function f and a number s,
such that f(x1)==f(x2) iff x1+x2=s of x1==x2
Our goal is to find such s
'''

def gaussian_elimination_mod2_with_rank(matrix):
    """
    Perform Gaussian elimination on a matrix in a 0,1 field (mod 2 arithmetic) and return its rank.

    Parameters:
    matrix (np.array): A numpy array representing the matrix.

    Returns:
    tuple: A tuple containing the rank of the matrix and the matrix after Gaussian elimination.
    """
    rows, cols = matrix.shape

    for i in range(min(rows, cols)):
        # Find a pivot for column i
        pivot_row = None
        for j in range(i, rows):
            if matrix[j, i] == 1:
                pivot_row = j
                break

        if pivot_row is None:
            continue

        # Swap pivot row into position
        matrix[[i, pivot_row]] = matrix[[pivot_row, i]]

        # Eliminate all other 1's in this column
        for j in range(rows):
            if j != i and matrix[j, i] == 1:
                matrix[j] = (matrix[j] + matrix[i]) % 2

    # Calculate the rank as the number of unique non-zero rows
    unique_rows = {tuple(row) for row in matrix if np.any(row)}
    rank = len(unique_rows)

    return rank, matrix





def find_non_trivial_solution(M):
    """
    Find a non-trivial solution to Mx = 0 in a 0,1 field.

    Parameters:
    M (np.array): A (n-1) x n matrix in a 0,1 field after Gaussian elimination with rank n-1.

    Returns:
    np.array: A non-trivial solution vector x.
    """
    rows, cols = M.shape

    # Generate all possible non-zero binary vectors of length cols
    for x in product([0, 1], repeat=cols):
        x = np.array(x)
        if np.any(x) and np.all(np.dot(M, x) % 2 == 0):
            return x

    return None


class Simon(QuantumAlgorithm):

    def __init__(self, num_qubits: int) -> None:
        if (num_qubits % 2) != 0:
            raise ValueError("The number of qubits must be even in Simon problem")
        super().__init__(num_qubits)
        self.circuit = Circuit.NumpyCircuit(num_qubits)
        self.num_qubits = num_qubits
        self._solution = -1
        self.uf = []
        self._s = 0

    def construct_circuit(self) -> None:
        '''
        For convenience, we use Allhadamard gats instead of half of the hadamard. This does not
        affect the result since we only measure the above half
        '''
        inputsize = self.num_qubits // 2
        for i in range(inputsize):
            self.circuit.add_gate(Gate.Hadamard(), i)
        self.compile_uf()
        for i in range(inputsize):
            self.circuit.add_gate(Gate.Hadamard(), i)

    def set_input(self, uf: List) -> None:
        func_dim = int(self.num_qubits / 2)
        if len(uf) != (1 << func_dim):
            raise ValueError("The input dimension does not match!")
        self._uf = uf

    def compile_uf(self) -> None:
        func_dim = self.num_qubits // 2
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
        func_dim = self.num_qubits // 2
        measure_indices = list(range(0, func_dim))
        y_result = []
        pre_rank = 0
        while not enough:
            self.circuit.clear_all()
            self.circuit.compute()
            result = self.circuit.measure(measure_indices)
            print(f"Generate a new y {result}")
            matrix = copy.copy(y_result)
            matrix.append(result)

            rank, gaus_matrix = gaussian_elimination_mod2_with_rank(np.array(matrix))

            #print(f"new rank= {rank}, marix is \n {matrix},gaus_matrix is  \n {gaus_matrix}")
            if rank > pre_rank:
                y_result.append(result)
                if len(y_result) == (func_dim - 1):
                    enough = True
                pre_rank = rank

        matrix = np.array(y_result)
        rank, gaus_matrix = gaussian_elimination_mod2_with_rank(np.array(matrix))
        print(matrix)
        s = find_non_trivial_solution(gaus_matrix)
        self._solution = convert_list_to_int(func_dim, list(s))
        print(f"The solution for the given function is s={s}")
        return

    @property
    def solution(self):
        return self._solution


class Simon_qiskit(QuantumAlgorithm):
    def __init__(self, num_qubits: int) -> None:
        if (num_qubits % 2) != 0:
            raise ValueError("The number of qubits must be even in Simon problem")
        super().__init__(num_qubits)
        self.circuit = qiskit.QuantumCircuit(num_qubits, num_qubits // 2)
        self.simulator = AerSimulator()
        self.num_qubits = num_qubits
        self._solution = -1
        self.uf = []
        self._s = 0

    def construct_circuit(self) -> None:
        '''
        For convenience, we use Allhadamard gats instead of half of the hadamard. This does not
        affect the result since we only measure the above half
        '''
        inputsize = self.num_qubits // 2
        for i in range(inputsize):
            self.circuit.h(i)
        self.compile_uf()
        for i in range(inputsize):
            self.circuit.h(i)
        self.circuit.measure(list(range(0, inputsize)), list(range(0, inputsize)))

    def set_input(self, uf: List) -> None:
        func_dim = int(self.num_qubits / 2)
        if len(uf) != (1 << func_dim):
            raise ValueError("The input dimension does not match!")
        self._uf = uf

    def compile_uf(self) -> None:
        func_dim = self.num_qubits // 2
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
                        self.circuit.cx(0, 1)
                    else:
                        for i in range(0, func_dim):
                            if input_bit[i] == 0:
                                self.circuit.x(i)
                        self.circuit.mcx(control_qubits=list(range(0, func_dim)),
                                         target_qubit=func_dim + j)
                        for i in range(0, func_dim):
                            if input_bit[i] == 0:
                                self.circuit.x(i)
        return

    def compute_result(self) -> None:
        enough = False
        func_dim = self.num_qubits // 2
        measure_indices = list(range(0, func_dim))
        y_result = []
        pre_rank = 0
        while not enough:
            compiled_circuit = qiskit.transpile(self.circuit, self.simulator)
            # Execute the circuit on the aer simulator
            job = self.simulator.run(compiled_circuit, shots=1)

            # Grab results from the job
            result = job.result()
            # Returns counts
            counts = result.get_counts(compiled_circuit)
            result = list(counts.keys())[0]
            result = result[::-1]
            result = [int(char) for char in result]
            #print(f"Generate a new y {result}")
            matrix = copy.copy(y_result)
            matrix.append(result)
            rank, gaus_matrix = gaussian_elimination_mod2_with_rank(np.array(matrix))

            #print(f"new rank= {rank}, marix is \n {matrix},gaus_matrix is  \n {gaus_matrix}")
            if rank > pre_rank:
                y_result.append(result)
                if len(y_result) == (func_dim - 1):
                    enough = True
                pre_rank = rank

        matrix = np.array(y_result)
        #print(matrix)
        rank, gaus_matrix = gaussian_elimination_mod2_with_rank(matrix)

        #print(f"Final rank= {rank}, marix is \n {matrix},gaus_matrix is  \n {gaus_matrix}")
        self._solution = find_non_trivial_solution(gaus_matrix)
        self._solution=convert_list_to_int(func_dim,list(self._solution))
        print(f"The solution for the given function is s={self._solution}")
        return

    @property
    def solution(self):
        return self._solution
