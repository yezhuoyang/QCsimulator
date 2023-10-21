import numpy as np

import Gate
import Parameter
from Gate import *
from Gate.Gate import QuantumGate
from State import QuantumState
from typing import List, Union, Any
import re

class DensityMatrix:

    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits

    def add_gate(self, gate: QuantumGate, qubit_indices: list[int]) -> None:
        raise NotImplementedError("Subclasses must implement add_gate method.")

    def add_noise(self, noise: QuantumNoise,qubit_indices: list[int]) -> None:
        raise NotImplementedError("Subclasses must implement add_gate method.")

    '''
    Implement the computation process of the whole circuit using density matrix.
    The state should be in the final state after evolving after the compute method
    is called
    '''
    def compute(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement compute method.")

    '''
    Measure a single qubit, after which the state is still a quantum state, 
    But the state of the measured qubit has collapsed to |0> or |1>
    '''
    def measure(self, qubit_index: int) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement measure method.")

    '''
    Calculate probability of a given state with all qubits at the same time.
    :statestr A string describing the state that you want to know the probability 
    For example, in a 2-qubit setting, the statestr can be "00" or "01", or "10" or "11"
    '''
    def measureAll(self, statestr: str) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement measure method.")

    '''
    Return the expectation value of the final quantum state given observable
    '''
    def expectation(self, observable: QuantumGate, qubit_indices: Union[list[int], int]) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement expectation method.")


class NumpyDensity(DensityMatrix):

    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits

    def add_gate(self, gate: QuantumGate, qubit_indices: list[int]) -> None:
        raise NotImplementedError("Subclasses must implement add_gate method.")

    def add_noise(self, noise: QuantumNoise,qubit_indices: list[int]) -> None:
        raise NotImplementedError("Subclasses must implement add_noise method.")

    '''
    Implement the computation process of the whole circuit using density matrix.
    The state should be in the final state after evolving after the compute method
    is called
    '''
    def compute(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement compute method.")

    '''
    Measure a single qubit, after which the state is still a quantum state, 
    But the state of the measured qubit has collapsed to |0> or |1>
    '''
    def measure(self, qubit_index: int) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement measure method.")

    '''
    Calculate probability of a given state with all qubits at the same time.
    :statestr A string describing the state that you want to know the probability 
    For example, in a 2-qubit setting, the statestr can be "00" or "01", or "10" or "11"
    '''
    def measureAll(self, statestr: str) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement measure method.")

    '''
    Return the expectation value of the final quantum state given observable
    '''
    def expectation(self, observable: QuantumGate, qubit_indices: Union[list[int], int]) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement expectation method.")



class TorchDensity(DensityMatrix):

    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits

    def add_gate(self, gate: QuantumGate, qubit_indices: list[int]) -> None:
        raise NotImplementedError("Subclasses must implement add_gate method.")

    def add_noise(self, noise: QuantumNoise,qubit_indices: list[int]) -> None:
        raise NotImplementedError("Subclasses must implement add_noise method.")

    '''
    Implement the computation process of the whole circuit using density matrix.
    The state should be in the final state after evolving after the compute method
    is called
    '''
    def compute(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement compute method.")

    '''
    Measure a single qubit, after which the state is still a quantum state, 
    But the state of the measured qubit has collapsed to |0> or |1>
    '''
    def measure(self, qubit_index: int) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement measure method.")

    '''
    Calculate probability of a given state with all qubits at the same time.
    :statestr A string describing the state that you want to know the probability 
    For example, in a 2-qubit setting, the statestr can be "00" or "01", or "10" or "11"
    '''
    def measureAll(self, statestr: str) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement measure method.")

    '''
    Return the expectation value of the final quantum state given observable
    '''
    def expectation(self, observable: QuantumGate, qubit_indices: Union[list[int], int]) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement expectation method.")


class GpuDensity(DensityMatrix):

    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits

    def add_gate(self, gate: QuantumGate, qubit_indices: list[int]) -> None:
        raise NotImplementedError("Subclasses must implement add_gate method.")

    def add_noise(self, noise: QuantumNoise,qubit_indices: list[int]) -> None:
        raise NotImplementedError("Subclasses must implement add_noise method.")

    '''
    Implement the computation process of the whole circuit using density matrix.
    The state should be in the final state after evolving after the compute method
    is called
    '''
    def compute(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement compute method.")

    '''
    Measure a single qubit, after which the state is still a quantum state, 
    But the state of the measured qubit has collapsed to |0> or |1>
    '''
    def measure(self, qubit_index: int) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement measure method.")

    '''
    Calculate probability of a given state with all qubits at the same time.
    :statestr A string describing the state that you want to know the probability 
    For example, in a 2-qubit setting, the statestr can be "00" or "01", or "10" or "11"
    '''
    def measureAll(self, statestr: str) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement measure method.")

    '''
    Return the expectation value of the final quantum state given observable
    '''
    def expectation(self, observable: QuantumGate, qubit_indices: Union[list[int], int]) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement expectation method.")
