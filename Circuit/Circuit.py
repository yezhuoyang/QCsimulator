import numpy as np

import Parameter
from Gate import *
from Gate.Gate import QuantumGate
from State import QuantumState
from typing import List, Union


class QuantumCircuit:
    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits

    def add_gate(self, gate: QuantumGate, qubit_indices: list[int]) -> None:
        raise NotImplementedError("Subclasses must implement add_gate method.")

    '''
    Implement the computation process of the whole circuit.
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

    def visulize(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement visulize method.")

    def transpile_qase(self, qasmStr: str) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement transpile_qase method.")

    def to_qasm(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement to_qasm method.")


'''
Quantum Circuit class simulated by Numpy and naive matrix/vector multiplication method
'''


class NumpyCircuit(QuantumCircuit):
    def __init__(self, num_qubits: int) -> None:
        super().__init__(num_qubits)
        '''
        By default, initial the input state as |000...0>
        '''
        self.state = QuantumState(qubit_number=num_qubits)
        self.gate_list = []
        self.gate_num = 0
        self.calc_dict = {item: False for item in self.gate_list}
        self.calc_step = 0

    '''
    Add Gate in sequence
    Each element is a tuple of the form: (Gate,indices)
    indices here must be a list
    '''

    def add_gate(self, gate: QuantumGate, qubit_indices: Union[list[int], int]) -> None:
        '''
        :param gate: A quantum gate, for example: Hadmard
        :param qubit_indices: The indices of qubit that the gate act on, for example
        If the hardmard is acted on the third qubit, it should be [3]
        If teh CNOT gate acted on the forth and fifth qubits, it should be [4,5]
        :return:
        '''
        '''
        First, check the dimension of qubit_indices
        '''
        if isinstance(qubit_indices, int):
            if gate.num_qubits != 1:
                raise ValueError("qubit_indices for multi-qubit gate has to be a List!")
        if isinstance(qubit_indices, List):
            if gate.num_qubits != len(qubit_indices):
                raise ValueError("The length of the qubit_indices list has to match with the qubit number of the gate!")
        self.gate_list.append((gate, qubit_indices, self.gate_num))
        self.gate_num += 1

    '''
    Change the index between two qubits of the input state_vector and return the modified state vector
    Params:
    '''

    def qubit_state_swap(self, num_qubits: int, state_vector: np.ndarray, index1: int, index2: int) -> np.ndarray:

        return

    '''
    Change the index between two qubits of the input gate_matrix and return the modified gatematrix
    Params:
    '''

    def qubit_matrix_swap(self, num_qubits: int, gate_matrix: np.ndarray, index1: int, index2: int) -> np.ndarray:

        return

    '''
    TODO:Calculate the matrix form of a gate after kron product
    This part is the most difficult because for a 2-qubit gates,
    we have to swap two index together before we can do simple kroneck product
    '''

    def calc_kron(self, gateindex: int) -> np.ndarray:
        (gate, qubit_indices, index) = self.gate_list[gateindex]
        I = np.array([[1, 0], [0, 1]], dtype=Parameter.qtype)
        if gate.num_qubits == 1:
            if gateindex == 0:
                matrix = gate.matrix()
            else:
                matrix = I
                for i in range(0, gateindex - 1):
                    matrix = np.kron(matrix, I)
            for i in range(gateindex, self.num_qubits):
                matrix = np.kron(matrix, I)
        else:
            '''
            TODO: Two or three qubit gates
            '''
            return None
        return matrix

    def compute(self) -> None:
        while self._compute_step():
            continue
        return

    '''
    Take on step in computation. Only do one matrix multiplication between a gate and 
    the quantum state.
    Return value: Return True if the function caclulate a gate. 
    If it does nothing, return False
    '''

    def _compute_step(self) -> bool:
        if self.calc_step == self.gate_num:
            return False
        '''
        Avoid repeated computation
        '''
        if self.calc_dict[self.gate_list[self.calc_step]]:
            return False
        '''
        First get the whole matrix after doing kron product
        '''
        matrix = self.calc_kron(self.calc_step)
        self.state = np.matmul(matrix, self.state)
        self.calc_dict[self.gate_list[self.calc_step]] = True
        self.calc_step += 1
        return True

    def measure(self, qubit_index: int) -> NotImplementedError:
        '''
        First make sure the computation is done
        '''
        if self.calc_step != self.num_qubits:
            self.compute()
        return NotImplementedError("Subclasses must implement measure method.")

    def measureAll(self, statestr: str) -> NotImplementedError:
        '''
        First make sure the computation is done
        '''
        if self.calc_step != self.num_qubits:
            self.compute()
        return NotImplementedError("Subclasses must implement measureAll method.")

    def visulize(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement visulize method.")

    def transpile_qase(self, qasmStr: str) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement transpile_qase method.")

    def to_qasm(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement to_qasm method.")


'''
Quantum Circuit Class simulated by Pytorch
'''


class PytorchCircuit(QuantumCircuit):

    def add_gate(self, gate: QuantumGate, qubit_indices: list[int]) -> None:
        raise NotImplementedError("Subclasses must implement add_gate method.")

    def compute(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement compute method.")

    def measure(self, qubit_index: int) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement measure method.")

    def visulize(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement visulize method.")

    def transpile_qase(self, qasmStr: str) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement transpile_qase method.")

    def to_qasm(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement to_qasm method.")


'''
Quantum Circuit Class simulated by GPU
'''


class GPUCircuit(QuantumCircuit):

    def add_gate(self, gate: QuantumGate, qubit_indices: list[int]) -> None:
        raise NotImplementedError("Subclasses must implement add_gate method.")

    def compute(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement compute method.")

    def measure(self, qubit_index: int) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement measure method.")

    def visulize(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement visulize method.")

    def transpile_qase(self, qasmStr: str) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement transpile_qase method.")

    def to_qasm(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement to_qasm method.")
