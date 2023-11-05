from typing import List

from Algorithm import QuantumAlgorithm

import Gate
import Parameter
import Circuit
from typing import List, Union, Any
import re


class BVAlgorithm(QuantumAlgorithm):

    def __init__(self, num_qubits: int) -> None:
        super().__init__(num_qubits)
        self.num_qubits = num_qubits
        self.circuit = Circuit.NumpyCircuit(num_qubits)
        self.computed = False
        self._a = 0
        self._b = 0

    '''
    The circuit structure is the same as DuetchJosa
    '''

    def construct_circuit(self) -> None:
        inputdim = self.num_qubits - 1
        '''
        The first layer of Hadmard 
        '''
        self.circuit.add_gate(Gate.PauliX(), [inputdim])
        self.circuit.add_gate(Gate.AllHadamard(self.num_qubits), list(range(0, self.num_qubits)))
        self.compile_func()
        self.circuit.add_gate(Gate.AllHadamard(self.num_qubits), list(range(0, self.num_qubits)))

    '''
    The input of Berstain vazirani is a linear function f(x)=ax+b.
    We are asked to calculate a,b.
    b is 0 or 1
    a is a n-bit number. a<=((1<<numberqubit-1)-1)
    '''

    def set_input(self, parameter: List) -> None:
        if len(parameter) != 2:
            raise ValueError("Berstain vazirani must have two input parameter a,b!")
        self._a = parameter[0]
        self._b = parameter[1]
        if self._b != 0 and self._b != 1:
            raise ValueError("b has to be 0 or 1")
        if not (self._a >= 0 and self._a < (1 << (self.num_qubits - 1))):
            raise ValueError("a out of range")

    def convert_int_to_list(self, alginput: int):
        controllist = []
        k = alginput
        for i in range(0, self.num_qubits - 1):
            controllist.insert(0, k % 2)
            k = (k >> 1)
        return controllist

    def compile_func(self) -> None:
        alist = self.convert_int_to_list(self._a)
        for i in range(0, self.num_qubits - 1):
            if alist[i] == 1:
                self.circuit.add_gate(Gate.CNOT(), [i, self.num_qubits - 1])
        if self._b==1:
            self.circuit.add_gate(Gate.PauliX(), [self.num_qubits-1])
        return

    def compute_result(self) -> None:
        self.circuit.compute()
        result = self.circuit.measure(list(range(0, self.num_qubits - 1)))
        self.computed = True
        print(f"The function is f(x)={result}x+{self._b}")
