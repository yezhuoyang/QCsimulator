from typing import List

from Algorithm import QuantumAlgorithm

import Gate
import Parameter
import Circuit
from typing import List, Union, Any
import re
from .util import convert_int_to_list, convert_list_to_int
import qiskit
import functools
from qiskit_aer import AerSimulator


class BVAlgorithm(QuantumAlgorithm):

    def __init__(self, num_qubits: int) -> None:
        super().__init__(num_qubits)
        self.num_qubits = num_qubits
        self.circuit = Circuit.NumpyCircuit(num_qubits)
        self.computed = False
        self._a = 0
        self._b = 0
        self.computed_a_value = -1

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
        if not (0 <= self._a < (1 << (self.num_qubits - 1))):
            raise ValueError("a out of range")

    def compile_func(self) -> None:
        alist = convert_int_to_list(self.num_qubits - 1, self._a)
        for i in range(0, self.num_qubits - 1):
            if alist[i] == 1:
                self.circuit.add_gate(Gate.CNOT(), [i, self.num_qubits - 1])
        if self._b == 1:
            self.circuit.add_gate(Gate.PauliX(), [self.num_qubits - 1])
        return

    def compute_result(self) -> None:
        self.circuit.compute()
        result = self.circuit.measure(list(range(0, self.num_qubits - 1)))
        self.computed = True
        self.computed_a_value = convert_list_to_int(self.num_qubits - 1, result)
        print(f"The function is f(x)={result}x+{self._b}")

    def a_result(self) -> int:
        return self.computed_a_value


class BVAlgorithm_qiskit(QuantumAlgorithm):

    def __init__(self, num_qubits: int) -> None:
        super().__init__(num_qubits)
        self.num_qubits = num_qubits
        self.circuit = qiskit.QuantumCircuit(num_qubits, num_qubits - 1)
        self.simulator = AerSimulator()
        self.computed = False
        self._a = 0
        self._b = 0
        self.computed_a_value = -1

    '''
    The circuit structure is the same as DuetchJosa
    '''

    def construct_circuit(self) -> None:
        inputdim = self.num_qubits - 1
        '''
        The first layer of Hadmard 
        '''
        self.circuit.x(inputdim)
        self.circuit.h(list(range(0, self.num_qubits)))
        self.compile_func()
        self.circuit.h(list(range(0, self.num_qubits)))
        self.circuit.measure(list(range(0, self.num_qubits - 1)), list(range(0, self.num_qubits - 1)))

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
        if not (0 <= self._a < (1 << (self.num_qubits - 1))):
            raise ValueError("a out of range")

    def compile_func(self) -> None:
        alist = convert_int_to_list(self.num_qubits - 1, self._a)
        for i in range(0, self.num_qubits - 1):
            if alist[i] == 1:
                self.circuit.cx(i, self.num_qubits - 1)
        if self._b == 1:
            self.circuit.x(self.num_qubits - 1)
        return

    def compute_result(self) -> None:
        compiled_circuit = qiskit.transpile(self.circuit, self.simulator)

        # Execute the circuit on the aer simulator
        job = self.simulator.run(compiled_circuit, shots=1)

        # Grab results from the job
        result = job.result()
        # print(result)
        # Returns counts
        counts = result.get_counts(compiled_circuit)
        result = list(counts.keys())[0]
        self.computed = True
        self.computed_a_value = int(result[::-1], 2)
        print(f"The function is f(x)={result[::-1]}x+{self._b}")

    def a_result(self) -> int:
        return self.computed_a_value
