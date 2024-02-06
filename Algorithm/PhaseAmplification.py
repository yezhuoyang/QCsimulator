import qiskit
import functools
from qiskit_aer import AerSimulator
from Algorithm import QuantumAlgorithm
import Gate
import Parameter
import Circuit
from typing import List, Union, Any




class PhaseAmplification(QuantumAlgorithm):
    def __init__(self, num_qubits: int) -> None:
        super().__init__(num_qubits)
        self.num_qubits = num_qubits

    def construct_circuit(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement construct_circuit method.")

    def set_input(self, alginput: List) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement set_input method.")

    def compute_result(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement compute_result method.")