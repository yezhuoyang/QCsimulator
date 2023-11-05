import numpy as np

import Gate
import Parameter
import Circuit
from typing import List, Union, Any
import re
from Algorithm import QuantumAlgorithm


class DuetchJosa(QuantumAlgorithm):

    def __init__(self, num_qubits) -> None:
        self.num_qubits = num_qubits
        self.circuit = Circuit.NumpyCircuit(num_qubits)
        self.UF = []

    def set_input(self, uf: List) -> None:
        self.UF = uf
        if not self.check_uf():
            raise ValueError("Uf is not a valid input")
        raise None

    '''
    Check weather uf is a legal input: Either balance or constant
    what's more, the size of the uz should be num_qubit -1 because we need 
    one more qubit to keep Uf unitary
    '''

    def check_uf(self) -> bool:
        if not len(self.UF) == (1 << (self.num_qubits - 1)):
            return False
        count = 0
        for i in range(0, len(self.UF)):
            if self.UF[i] == 0:
                count += 1
        if count == (len(self.UF)) or count == 0:
            return True
        if count == (1 << (self.num_qubits - 2)):
            return True
        return False

    def construct_circuit(self):
        inputdim = self.num_qubits - 1
        '''
        The first layer of Hadmard 
        '''
        for i in range(0, inputdim):
            self.circuit.add_gate(Gate.Hadamard(), i)
        



        raise NotImplementedError("Subclasses must implement construct_circuit method.")

    def compute_result(self) -> NotImplementedError:
        raise NotImplementedError("Subclasses must implement compute_result method.")

    '''
    Compiler the Uf gate given the List of Uf input
    '''

    def compile_uf(self):

        return
