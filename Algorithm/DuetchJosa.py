import numpy as np

import Gate
import Parameter
import Circuit
from typing import List, Union, Any
import re
from Algorithm import QuantumAlgorithm


class DuetchJosa(QuantumAlgorithm):

    def __init__(self, num_qubits: int) -> None:
        super().__init__(num_qubits)
        self.circuit = Circuit.NumpyCircuit(num_qubits)
        self.UF = []
        self.computed = False
        self.balance = False

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

    def construct_circuit(self) -> None:
        inputdim = self.num_qubits - 1
        '''
        The first layer of Hadmard 
        '''
        self.circuit.add_gate(Gate.PauliX(), [inputdim])
        self.circuit.add_gate(Gate.AllHadamard(self.num_qubits), list(range(0, self.num_qubits)))
        self.compile_uf()
        self.circuit.add_gate(Gate.AllHadamard(self.num_qubits), list(range(0, self.num_qubits)))

    def compute_result(self) -> None:
        self.circuit.compute()
        result = self.circuit.measure(list(range(0, self.num_qubits - 1)))
        if sum(result) == 0:
            self.balance = False
            print("The function is constant")
        else:
            self.balance = True
            print("The function is balanced")

    '''
    Compile the Uf gate given the List of Uf input
    We should use MultiControlX gate here for convenience
    '''

    def compile_uf(self) -> None:
        for i in range(0, 1 << self.num_qubits - 1):
            if self.UF[i] == 1:
                self.circuit.add_gate(Gate.MultiControlX(self.num_qubits, self.convert_int_to_list(i)),
                                      list(range(0, self.num_qubits)))
        return

    def convert_int_to_list(self, alginput: int):
        controllist = []
        k = alginput
        for i in range(0, self.num_qubits - 1):
            controllist.insert(0, k % 2)
            k = (k >> 1)
        return controllist
