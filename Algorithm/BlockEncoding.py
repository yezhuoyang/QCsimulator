import numpy as np

from Algorithm import QuantumAlgorithm
import Gate
import Parameter
import Circuit
from typing import List, Union, Any
import qiskit
import qiskit.quantum_info as qi
from qiskit.circuit.library.standard_gates import XGate, RYGate




def norm_2_distance(matrix1: np.ndarray, matrix2: np.ndarray):
    return np.sum((matrix1 - matrix2) ** 2)


'''
The block encoding class for BCM matrix
'''


class BlockEncoding_qiskit(QuantumAlgorithm):
    def __init__(self, m: int, n: int):
        self.num_qubits = m + n + 1
        self._n = n
        self._m = 2
        super().__init__(self.num_qubits)
        self.circuit = qiskit.QuantumCircuit(self.num_qubits, m + 1)
        self._constructed = False

    '''
    The input of the BCM block encoding is alpha,beta,gamma
    '''

    def set_input(self, input: List):
        self._alpha = input[0]
        self._beta = input[1]
        self._gamma = input[2]

    def construct_circuit(self):
        if self._constructed:
            return
        self.construct_diffusion()
        self.construct_OA()
        self.construct_Oc()
        self.construct_diffusion()
        # self.circuit.measure(list(range(self._n, self._n + self._m + 1)), list(range(0, self._m + 1)))
        self._constructed = True
        return

    '''
    Return the 2**n * 2**n size A, which has been
    encoded in the left top of U
    '''

    def get_encoded_matrix(self):
        op = qi.Operator(self.circuit)
        return np.real(np.array(op)[0:2 ** self._n, 0:2 ** self._n])

    def construct_OA(self):
        theta0 = 2 * np.arccos(self._gamma)
        rot0 = RYGate(theta0).control(2, ctrl_state='00')
        self.circuit.append(rot0, [self._n + self._m - 1, self._n + self._m - 2, self._n + self._m])
        theta1 = 2 * np.arccos(self._alpha - 1)
        rot1 = RYGate(theta1).control(2, ctrl_state='01')
        self.circuit.append(rot1, [self._n + self._m - 1, self._n + self._m - 2, self._n + self._m])
        theta2 = 2 * np.arccos(self._beta)
        rot2 = RYGate(theta2).control(2, ctrl_state='10')
        self.circuit.append(rot2, [self._n + self._m - 1, self._n + self._m - 2, self._n + self._m])

    def construct_Oc(self):
        circuit_L = self.get_L_circuit()
        circuit_R = self.get_R_circuit()
        control_indices = [self._n + 1, self._n]
        gate_indices = list(range(0, self._n))
        gate_L = circuit_L.to_gate().control(2, ctrl_state='10')
        gate_R = circuit_R.to_gate().control(2, ctrl_state='00')
        self.circuit.append(gate_R, control_indices + gate_indices)
        self.circuit.append(gate_L, control_indices + gate_indices)
        return

    def construct_diffusion(self):
        self.circuit.h(list(range(self._n, self._n + self._m)))

    def get_L_circuit(self):
        circuit_L = qiskit.QuantumCircuit(self._n)
        for index in range(0, self._n - 1):
            control_qubits = list(range(0, self._n - 1 - index))
            target_qubit = self._n - 1 - index
            circuit_L.mcx(control_qubits,
                          target_qubit=[target_qubit])
        circuit_L.x(0)
        return circuit_L

    def get_R_circuit(self):
        circuit_R = qiskit.QuantumCircuit(self._n)
        for index in range(0, self._n - 1):
            num_control = self._n - 1 - index
            control_qubits = list(range(0, num_control))
            target_qubit = self._n - 1 - index

            cccx = XGate().control(num_ctrl_qubits=self._n - 1 - index, ctrl_state='0' * num_control)
            circuit_R.append(cccx, control_qubits + [target_qubit])
        circuit_R.x(0)
        return circuit_R

    def get_BCM(self):
        size = 2 ** self._n
        A = np.zeros(
            (size, size), dtype=np.float64
        )
        np.fill_diagonal(A, self._alpha)
        for i in range(size - 1):
            A[i, i + 1] = self._gamma
            A[i + 1, i] = self._beta
        A[0, size - 1] = self._beta
        A[size - 1, 0] = self._gamma
        return A / (2 ** self._m)

    def compute_result(self):
        self.construct_circuit()
        return self.get_encoded_matrix()

    '''
    Return the accuracy for block encoding
    '''

    def accuracy(self):
        return 1-norm_2_distance(self.compute_result(),self.get_BCM())


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
    BCM3 = BlockEncoding_qiskit(2, 3)
    BCM3.set_input([0.2, 0.3, 0.4])
    print("------------The target matrix----------------------------")
    print(BCM3.get_BCM())
    final_matrix_BC3 = BCM3.compute_result()
    print("--------------The BCM3(0.2,0.3,0.4) result--------------------------")
    print(final_matrix_BC3)
    print("--------------Fidelity calculated by matrix 2 norm-------------------")
    print(BCM3.accuracy())
    print("---------------The end-------------------------")




    BC4 = BlockEncoding_qiskit(2, 4)
    BC4.set_input([0.2, 0.3, 0.4])
    print("------------The target matrix----------------------------")
    print(BC4.get_BCM())
    final_matrix_BC4 = BC4.compute_result()
    print("--------------The BCM4(0.2,0.3,0.4) result--------------------------")
    print(final_matrix_BC4)
    print("--------------Fidelity calculated by matrix 2 norm-------------------")
    print(BC4.accuracy())
    print("---------------The end-------------------------")