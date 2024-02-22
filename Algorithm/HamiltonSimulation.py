from Algorithm import QuantumAlgorithm
import Gate
import Parameter
import Circuit
from typing import List, Union, Any
import qiskit
import qiskit.quantum_info as qi
from qiskit.circuit.library.standard_gates import XGate, RYGate, CXGate, RZGate, HGate, ZGate
from qiskit.circuit.library import UnitaryGate
import numpy as np
import scipy

'''
Implementation of Hamiltonian simulation 
'''


def norm_2_distance(matrix1: np.ndarray, matrix2: np.ndarray):
    return np.sum((matrix1 - matrix2) ** 2)


class HamiltonianSimulation(QuantumAlgorithm):
    def __init__(self, num_qubits: int) -> None:
        super().__init__(num_qubits)
        self._num_qubits = num_qubits

    def construct_circuit(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement construct_circuit method.")

    def set_input(self, alginput: List) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement set_input method.")

    def compute_result(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement compute_result method.")


'''
Simulate the hamiltonian 
   H=\sum_{k=0}^m Z_iZ_{i+1}+X_i
'''


class HamiltonianSimulation_ZZX_qiskit(QuantumAlgorithm):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self._num_qubits = num_qubits
        self.circuit = qiskit.QuantumCircuit(self._num_qubits, self._num_qubits)
        self.Hcircuit = qiskit.QuantumCircuit(self._num_qubits, self._num_qubits)
        self._constructed = False
        self._glist = [1] * num_qubits
        self._step = 10
        self._evolve_time = 0

    def construct_circuit(self):
        Xmatrix = np.array([[0, 1],
                            [1, 0]])
        Zmatrix = np.array([[1, 0],
                            [0, -1]])
        for qindex in range(0, self._num_qubits - 1):
            self.Hcircuit.append(UnitaryGate(-1 * Zmatrix), [qindex + 1])
            self.Hcircuit.append(ZGate(), [qindex + 1])
            self.Hcircuit.append(UnitaryGate(self._glist[qindex] * Xmatrix), [qindex])

        for i in range(0, self._step):
            for qindex in range(0, self._num_qubits - 1):
                self.construct_ZZ(qindex)
                self.construct_X(qindex)

    def construct_ZZ(self, qindex: int):
        self.circuit.append(CXGate(), [qindex + 1, qindex])
        self.circuit.append(RZGate(-2 * self._evolve_time / self._step), [qindex])
        self.circuit.append(CXGate(), [qindex + 1, qindex])
        return

    def construct_X(self, qindex: int):
        self.circuit.append(HGate(), [qindex])
        self.circuit.append(RZGate(2 * self._evolve_time * self._glist[qindex] / self._step), [qindex])
        self.circuit.append(HGate(), [qindex])
        return

    '''
    The input for g_i
    '''

    def set_input(self, g_list: List, step: int, evolve_time: float):
        assert len(g_list) == self._num_qubits
        self._glist = g_list
        self._step = step
        self._evolve_time = evolve_time

    '''
    Calculate the exact unitary.
    Use taylor expansion and truncate at order 1
    '''

    def get_exact_unitary(self, order: int):
        op = qi.Operator(self.Hcircuit)
        Hmatrix = np.array(op)
        coefficient = 1
        temp_mat = np.identity(2 ** self._num_qubits, dtype=complex)
        umatrix = np.zeros((2 ** self._num_qubits, 2 ** self._num_qubits), dtype=complex)
        for i in range(0, order):
            umatrix = umatrix + temp_mat * coefficient
            temp_mat = np.matmul(temp_mat, Hmatrix)
            coefficient = coefficient * (-1j) / (i + 1) * self._evolve_time
        return umatrix

    def compute_result(self):
        op = qi.Operator(self.circuit)
        return np.array(op)

    def norm2_distance(self):
        return norm_2_distance()


if __name__ == "__main__":
    XXZ = HamiltonianSimulation_ZZX_qiskit(2)
    XXZ.set_input([1, 1], 2, 1)
    XXZ.construct_circuit()
    print(XXZ.get_exact_unitary(10))

    print(XXZ.compute_result())
