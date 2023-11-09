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


def gaussian_elimination_mod2(matrix):
    num_rows, num_cols = matrix.shape

    # Perform Gaussian elimination to get an upper triangular matrix
    for col in range(num_cols):
        pivot_row = None
        for row in range(col, num_rows):
            if matrix[row][col] == 1:
                pivot_row = row
                break

        if pivot_row is None:
            continue

        matrix[[col, pivot_row]] = matrix[[pivot_row, col]]

        # Zero out all other entries in this column
        for i in range(num_rows):
            if i != col and matrix[i][col] == 1:
                matrix[i] = (matrix[i] + matrix[col]) % 2

    # Back substitution to find the solution vector 's'
    s = np.zeros(num_cols, dtype=int)  # Initialize solution vector with zeros

    # Start back substitution
    for row in range(num_rows - 1, -1, -1):
        pivot_cols = np.where(matrix[row] == 1)[0]
        if len(pivot_cols) == 1:
            # This means we have a pivot, so we can solve for this variable
            s[pivot_cols[0]] = 1
            # Zero out above entries
            for upper_row in range(row):
                if matrix[upper_row][pivot_cols[0]] == 1:
                    matrix[upper_row] = (matrix[upper_row] + matrix[row]) % 2
    print("After gaussion")
    print(matrix)
    return s


def gf2_rank(matrix):
    """
    Return the rank of a matrix in GF(2), which is the number of non-zero rows after
    row reduction.
    """
    A = np.array(matrix, dtype=np.int64) % 2
    num_rows, num_cols = A.shape

    row, col = 0, 0
    while row < num_rows and col < num_cols:
        # Find the pivot row and swap
        for pivot_row in range(row, num_rows):
            if A[pivot_row, col]:
                A[[row, pivot_row]] = A[[pivot_row, row]]
                break
        else:
            col += 1
            continue

        # Eliminate the column entries below the pivot
        for lower_row in range(row + 1, num_rows):
            if A[lower_row, col]:
                A[lower_row] ^= A[row]

        row += 1
        col += 1

    # The rank is the number of non-zero rows
    rank = np.count_nonzero(np.any(A, axis=1))
    return rank


def is_linearly_independent_mod2(vectors):
    """
    Check if a list of binary vectors is linearly independent in GF(2).
    :param vectors: a list of binary vectors
    :return: True if the list is linearly independent, False otherwise
    """
    matrix = np.array(vectors)
    rank = gf2_rank(matrix)
    return rank == len(vectors)


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
            if is_linearly_independent_mod2(matrix):
                y_result.append(result)
                if len(y_result) == (func_dim - 1):
                    enough = True

        matrix = np.array(y_result)
        print(matrix)
        s = gaussian_elimination_mod2(matrix)
        self._solution = convert_list_to_int(func_dim, s)
        print(f"The solution for the given function is s={s}")
        return

    @property
    def solution(self):
        return self._solution
