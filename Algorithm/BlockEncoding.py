import numpy as np

from Algorithm import QuantumAlgorithm
import Gate
import Parameter
import Circuit
from typing import List, Union, Any


class BlockEncoding_qiskit(QuantumAlgorithm):
    def __init__(self, num_qubits: int):
        super().__init__(num_qubits)
        self.num_qubits = num_qubits

    def construct_circuit(self):
        return


    '''
    The input of the block encoding is a 2-d matrix
    '''
    def set_input(self, matrix:np.ndarray):
        return

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