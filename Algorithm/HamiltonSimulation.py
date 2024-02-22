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
        self._constructed = False
        self._glist = [1] * num_qubits
        self._step = 10
        self._evolve_time = 0

    def construct_circuit(self):
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
        Xmatrix = np.array([[0, 1],
                            [1, 0]])
        Zmatrix = np.array([[1, 0],
                            [0, -1]])
        Hmatrix=np.zeros((2 ** self._num_qubits, 2 ** self._num_qubits), dtype=complex)
        for qindex in range(0, self._num_qubits - 1):
            Hcircuit=qiskit.QuantumCircuit(self._num_qubits, self._num_qubits)
            Hcircuit.append(UnitaryGate(-1 * Zmatrix), [qindex + 1])
            Hcircuit.append(ZGate(), [qindex])
            op_zz = np.array(qi.Operator(Hcircuit))
            Hcircuit = qiskit.QuantumCircuit(self._num_qubits, self._num_qubits)
            Hcircuit.append(UnitaryGate(self._glist[qindex] * Xmatrix), [qindex])
            op_x = np.array(qi.Operator(Hcircuit))
            Hmatrix=Hmatrix+op_zz+op_x
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
    step_list = list(range(1, 20, 1))
    distance_list_2 = []
    distance_list_4 = []
    distance_list_6 = []
    distance_list_8 = []
    distance_list_10 = []

    for step in step_list:
        XXZ = HamiltonianSimulation_ZZX_qiskit(2)
        XXZ.set_input([1, 1], step, 1)
        XXZ.construct_circuit()
        XXZ.get_exact_unitary(100)
        print("Step%d" % step)
        distance = XXZ.get_2_norm()
        distance_list_2.append(distance)
        print(distance)

    for step in step_list:
        XXZ = HamiltonianSimulation_ZZX_qiskit(4)
        XXZ.set_input([1, 1, 1, 1], step, 1)
        XXZ.construct_circuit()
        XXZ.get_exact_unitary(100)
        print("Step%d" % step)
        distance = XXZ.get_2_norm()
        distance_list_4.append(distance)
        print(distance)

    for step in step_list:
        XXZ = HamiltonianSimulation_ZZX_qiskit(6)
        XXZ.set_input([1, 1, 1, 1, 1, 1], step, 1)
        XXZ.construct_circuit()
        XXZ.get_exact_unitary(100)
        print("Step%d" % step)
        distance = XXZ.get_2_norm()
        distance_list_6.append(distance)
        print(distance)

    for step in step_list:
        XXZ = HamiltonianSimulation_ZZX_qiskit(8)
        XXZ.set_input([1, 1, 1, 1, 1, 1, 1, 1], step, 1)
        XXZ.construct_circuit()
        XXZ.get_exact_unitary(100)
        print("Step%d" % step)
        distance = XXZ.get_2_norm()
        distance_list_8.append(distance)
        print(distance)

    for step in step_list:
        XXZ = HamiltonianSimulation_ZZX_qiskit(10)
        XXZ.set_input([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], step, 1)
        XXZ.construct_circuit()
        XXZ.get_exact_unitary(100)
        print("Step%d" % step)
        distance = XXZ.get_2_norm()
        distance_list_8.append(distance)
        print(distance)
    '''
    Plot the result
    '''
    import matplotlib.pyplot as plt
    plt.scatter(step_list, distance_list_2, label="2 qubit ZZX H model")
    plt.plot(step_list, distance_list_2)

    plt.scatter(step_list, distance_list_4, label="4 qubit ZZX H model")
    plt.plot(step_list, distance_list_4)

    plt.scatter(step_list, distance_list_6, label="6 qubit ZZX H model")
    plt.plot(step_list, distance_list_6)

    plt.scatter(step_list, distance_list_8, label="8 qubit ZZX H model")
    plt.plot(step_list, distance_list_8)

    plt.scatter(step_list, distance_list_10, label="10 qubit ZZX H model")
    plt.plot(step_list, distance_list_10)

    plt.title("2-norm distance VS step size of Hamiltonian simulation")
    plt.xticks(step_list)
    plt.xlabel("Step size")
    plt.ylabel("2-Norm error")
    plt.grid(True)
    plt.legend()
    plt.savefig("2NormCompare.png")



