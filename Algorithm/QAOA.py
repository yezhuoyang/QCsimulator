from typing import List

from Algorithm import QuantumAlgorithm
import qiskit
from qiskit_aer import AerSimulator


class QAOA(QuantumAlgorithm):

    def __init__(self, num_qubits: int) -> None:
        super().__init__(num_qubits)
        self.num_qubits = num_qubits

    def construct_circuit(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement construct_circuit method.")

    def set_input(self, alginput: List) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement set_input method.")

    def compute_result(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement compute_result method.")


class QAOA_qiskit(QuantumAlgorithm):

    def __init__(self, num_qubits: int, iteration: int) -> None:
        super().__init__(num_qubits)
        '''
        The number of iteration(Mixer and cost hamiltonion)
        Every iteration will add twp more parameters to be optimized
        '''
        self._iteration = iteration
        '''
        Store all 2*iteration parameter
        '''
        self._parameters_mixer = [0]*iteration
        self._parameters_cost  = [0]*iteration
        self.num_qubits = num_qubits
        self.circuit = qiskit.QuantumCircuit(num_qubits, num_qubits - 1)
        self._simulator = AerSimulator()

    def construct_circuit(self) -> None:
        self.circuit.h(list(range(0, self.num_qubits)))
        '''
        Construct mixer and cost hamiltonion
        '''
        for i in range(self._iteration):
            self.construct_mixer(self._parameters_mixer[i])
            self.construct_ansatz(self._parameters_cost[i])
        return

    def construct_mixer(self, alpha):
        return

    def construct_ansatz(self, beta):
        return

    def set_parameter(self):
        return

    def set_optimizer(self):
        return

    def optimizer(self):
        return

    def set_input(self, alginput: List) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement set_input method.")

    def compute_result(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement compute_result method.")
