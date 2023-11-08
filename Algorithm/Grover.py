'''
Algorithmic framework of Grover algorithm
'''
from typing import List

from Algorithm import QuantumAlgorithm
import Circuit
import Gate
from util import convert_int_to_list, convert_list_to_int


class Grover(QuantumAlgorithm):

    def __init__(self, num_qubits: int) -> None:
        super().__init__(num_qubits)
        self.computed = False
        self.circuit = Circuit.NumpyCircuit(num_qubits)
        self.num_qubits = num_qubits
        '''
        How many time we may need to call the grover operator,
        initially set to be 1
        '''
        self.grover_step = 1
        self.max_step=10
        self._database = []
        self._solution=-1
        self._succeed=False

    def construct_circuit(self) -> None:
        self.circuit.add_gate(Gate.AllHadamard(self.num_qubits), list(range(0, self.num_qubits)))
        '''
        Construct grover circuit many times
        '''
        for i in range(self.grover_step):
            self.construct_grover_op()
        return

    '''
    zf add a (-1) phase to all states |x> that satisfies f(x)=1 
    '''

    def construct_zf(self):
        for i in range(0, len(self._database)):
            if self._database[i] == 1:
                '''
                When there are only two qubits, we can simply use
                controlZ gate
                '''
                if self.num_qubits == 2:
                    if i == 1:
                        self.circuit.add_gate(Gate.ControlledZ(), [0, 1])
                    elif i == 0:
                        self.circuit.add_gate(Gate.PauliX(), [0])
                        self.circuit.add_gate(Gate.ControlledZ(), [0, 1])
                        self.circuit.add_gate(Gate.PauliX(), [0])
                else:
                    self.circuit.add_gate(
                        Gate.MultiControlZ(self.num_qubits, convert_int_to_list(self.num_qubits - 1, i)),
                        [list(range(0, self.num_qubits - 1)), self.num_qubits - 1])

        return

    '''
    zo add a (-1) phase to all states |00...00>
    '''

    def construct_zo(self):
        if self.num_qubits == 2:
            self.circuit.add_gate(Gate.PauliX(), [0])
            self.circuit.add_gate(Gate.ControlledZ(), [0, 1])
            self.circuit.add_gate(Gate.PauliX(), [0])
        else:
            self.circuit.add_gate(
                Gate.MultiControlZ(self.num_qubits, [0] * (self.num_qubits - 1)),
                [list(range(0, self.num_qubits - 1)), self.num_qubits - 1])
        return

    def construct_grover_op(self):
        self.construct_zf()
        self.circuit.add_gate(Gate.AllHadamard(self.num_qubits), list(range(0, self.num_qubits)))
        self.construct_zo()
        self.circuit.add_gate(Gate.AllHadamard(self.num_qubits), list(range(0, self.num_qubits)))
        return

    '''
    The input of grover is the value stored in the 
    unstructured database
    '''

    def set_input(self, database: List) -> None:
        if len(database) != (1 << (self.num_qubits - 1)):
            raise ValueError("The size of the database doesn't match the qubit number")
        self._database = database

    '''
    Complete grover's algorithm, once we find a solution, return 
    Otherwise we add the grover step and do the algorithm again
    '''

    def compute_result(self) -> None:
        while not self.computed:
            self.construct_circuit()
            result = self.circuit.measure(list(range(0, self.num_qubits - 1)))
            result = convert_list_to_int(self.num_qubits - 1, result)
            if self._database[result] == 1:
                self.computed = True
                self._succeed=True
                self._solution=result
                print(f"Found a solution {result}")
                return
            else:
                self.grover_step=self.grover_step+1
                if self.grover_step>=self.max_step:
                    break
        if not self.computed:
            print("Algorithm failed!")
