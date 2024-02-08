
from Algorithm import QuantumAlgorithm
import Gate
import Parameter
import Circuit
from typing import List, Union, Any
'''
Implementation of Hamiltonian simulation 
'''



class HamiltonianSimulation(QuantumAlgorithm):
    def __init__(self, num_qubits: int) -> None:
        super().__init__(num_qubits)
        self.num_qubits = num_qubits

    def construct_circuit(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement construct_circuit method.")

    def set_input(self, alginput: List) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement set_input method.")

    def compute_result(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement compute_result method.")


class HamiltonianSimulation_qiskit(QuantumAlgorithm):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self.num_qubits = num_qubits

    def construct_circuit(self):
        return

    def set_input(self, alginput: List):
        return

    def compute_result(self):
        return