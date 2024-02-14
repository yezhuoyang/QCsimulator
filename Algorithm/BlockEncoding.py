import numpy as np

from Algorithm import QuantumAlgorithm
import Gate
import Parameter
import Circuit
from typing import List, Union, Any
import qiskit
import qiskit.quantum_info as qi
from qiskit.circuit.library.standard_gates import XGate

'''
The block encoding class for BCM matrix
'''


class BlockEncoding_qiskit(QuantumAlgorithm):
    def __init__(self, m: int, n: int):
        self.num_qubits = m + n + 1
        self._n = n
        self._m = m
        super().__init__(self.num_qubits)
        self.circuit = qiskit.QuantumCircuit(self.num_qubits, m + 1)


    '''
    The input of the BCM block encoding is alpha,beta,gamma
    '''

    def set_input(self, alpha: float, beta: float, gamma: float):
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma

    def construct_circuit(self):
        self.construct_diffusion()
        self.construct_OA()
        self.construct_Oc()
        self.construct_diffusion()
        self.circuit.measure(list(range(0, self._m + 1)), list(range(0, self._m + 1)))
        return

    '''
    Return the 2**n * 2**n size A, which has been
    encoded in the left top of U
    '''

    def get_encoded_matrix(self):
        op = qi.Operator(self.circuit)
        return np.array(op)[0:2 ** self._n, 0:2 ** self._n]

    def construct_OA(self):
        return

    def construct_Oc(self):
        return

    def construct_diffusion(self):
        self.circuit.h(list(range(1, self._m + 1)))

    def construct_L(self):
        circuit_L = qiskit.QuantumCircuit(self._n)
        for index in range(0, self._n - 1):
            control_qubits = list(range(0, self._n - 1 - index))
            target_qubit = self._n - 1 - index

            circuit_L.mcx(control_qubits,
                             target_qubit=[target_qubit])
        circuit_L.x(0)
        return circuit_L

    def construct_R(self):
        circuit_R = qiskit.QuantumCircuit(self._n)
        for index in range(0, self._n - 1):
            num_control = self._n - 1 - index
            control_qubits = list(range(0, num_control))
            target_qubit = self._n - 1 - index

            cccx = XGate().control(num_ctrl_qubits=self._n - 1 - index, ctrl_state='0'*num_control)
            circuit_R.append(cccx,control_qubits+[target_qubit])
        circuit_R.x(0)
        return circuit_R

    def get_BCM(self):
        size = 2 ** self._n
        A = np.zeros(
            (size, size), dtype=np.float64
        )
        np.fill_diagonal(A, self._alpha)
        for i in range(size - 1):
            A[i, i + 1] = self._beta
            A[i + 1, i] = self._gamma
        A[0, size - 1] = self._gamma
        A[size - 1, 0] = self._beta
        return A

    def compute_result(self):
        return

    '''
    Return the accuracy for block encoding
    '''

    def accuracy(self):
        return 0


class BlockEncoding(QuantumAlgorithm):
    def __init__(self, num_qubits: int) -> None:
        super().__init__(num_qubits)
        self.num_qubits = num_qubits

    def construct_circuit(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement construct_circuit method.")

    def set_input(self, alginput: List) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement set_input method.")

    def compute_result(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement compute_result method.")


if __name__ == "__main__":
    BC = BlockEncoding_qiskit(3, 2)

    circuit_L=BC.construct_L()
    circuit_R=BC.construct_R()
    op = qi.Operator(circuit_L)
    print(np.array(op))

    circuit_R=BC.construct_R()
    op = qi.Operator(circuit_R)
    print(np.array(op))

