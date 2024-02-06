import numpy as np

import Gate
import Parameter
from Gate import *
from Gate.Gate import QuantumGate
from State import QuantumState
from typing import List, Union
import re
from .Circuit import QuantumCircuit

'''
Use a python dictionary to store the circuit.
'''


class StateDictCircuit(QuantumCircuit):

    def __init__(self, num_qubits: int) -> None:
        super().__init__(num_qubits)
        '''
        By default, initial the input state as |000...0>
        '''
        self.state = QuantumState(qubit_number=num_qubits)
        self.gate_list = []
        '''
        The optimized calculation sequence, which should be a list of gate list
        For example:
              self.calc_sequence=[[Hadmard(0),Hadamard(1)],[CNOT(0,1)],[Hadmard(0),Hadamard(1)]]
              The calc_sequence can be optimized by the optimizer class before the actual calculation
        '''
        self.calc_sequence = []
        self.gate_num = 0
        self.calc_step = 0
        self.Debug = False
        '''
        Threshold for deciding whether to discard a state or not
        '''
        self.threshold = 1e-8
        '''
        Initialize the state
        The key is the n bit integer denoting the state of the qubit
        The value is the amplitude of this state 
        '''
        self._statedict = {0: Parameter.qtype(1)}

    '''
    Return True if the k^th bit of the stateint is 1
    For example, bit_is_1(2, 0b1011) is true because the third element is 1
    '''

    def bit_is_1(self, k: int, stateint: int) -> bool:
        return stateint == (stateint | (1 << (self.num_qubits - k - 1)))

    '''
    Flip the k^th bit of the given stateint
    For example, flip_bit(2,0b1011) will return 1001
    '''

    def flip_bit(self, k: int, stateint: int) -> int:
        return stateint ^ (1 << (self.num_qubits - k - 1))

    def add_gate(self, gate: QuantumGate, qubit_indices: list[int]) -> None:
        '''
        :param gate: A quantum gate, for example: Hadmard
        :param qubit_indices: The indices of qubit that the gate act on, for example
        If the hardmard is acted on the third qubit, it should be [3]
        If the CNOT gate acted on the forth and fifth qubits, it should be [4,5]
        If the multi controlled Z gate, controlled by the first and second qubit, add on the fifth qubit, it should be [[1,2],5]
        :return:
        '''
        '''
        First, check the dimension of qubit_indices
        '''
        if isinstance(qubit_indices, int):
            if gate.num_qubits != 1:
                raise ValueError("qubit_indices for multi-qubit gate has to be a List!")
            if qubit_indices < 0 or qubit_indices >= self.num_qubits:
                raise ValueError(f"qubit_indices {qubit_indices} out of range!")
            '''
            The step is to make sure it can be converted to tuple in the fowllowing steps.
            '''
            qubit_indices = [qubit_indices]
        elif isinstance(qubit_indices, List):
            if isinstance(gate, MultiControlX | MultiControlZ):
                if not isinstance(qubit_indices[0], List):
                    raise ValueError(
                        "The first element of qubit_indices for multi control qubit gate has to be a List!")
                if not len(qubit_indices[0]) == (gate.num_qubits - 1):
                    raise ValueError("The qubit number of multi control qubit doesn't match with the qubit_indices")
                for index in qubit_indices[0]:
                    if index < 0 or index >= self.num_qubits:
                        raise ValueError(f"{index} in qubit_indices {qubit_indices} out of range!")
                    if qubit_indices[1] < 0 or qubit_indices[1] > self.num_qubits:
                        raise ValueError(f"{qubit_indices[1]} in qubit_indices {qubit_indices} out of range!")
            else:
                if gate.num_qubits != len(qubit_indices):
                    raise ValueError(
                        "The length of the qubit_indices list has to match with the qubit number of the gate!")
                for index in qubit_indices:
                    if index < 0 or index >= self.num_qubits:
                        raise ValueError(f"{index} in qubit_indices {qubit_indices} out of range!")
            '''Make sure that the qubit indices of single qubit gate be just an integer'''
            if gate.num_qubits == 1:
                qubit_indices = qubit_indices[0]
        if isinstance(qubit_indices, int):
            qubit_indices = [qubit_indices]
        self.gate_list.append((gate, tuple(qubit_indices), self.gate_num))
        self.calc_sequence.append([(gate, tuple(qubit_indices), self.gate_num)])
        self.gate_num += 1

    @staticmethod
    def filter_small_value(keyvalue: tuple) -> bool:
        key, value = keyvalue
        return abs(value) > 0

    '''
    Convert the dictionary expression to numpy array
    '''

    def convert_state_dict_to_array(self) -> np.ndarray:
        state_vector = np.array([0] * (2 ** self.num_qubits), dtype=Parameter.qtype)
        for key in self._statedict.keys():
            state_vector[key] = Parameter.qtype(self._statedict[key])
        return state_vector

    def compute(self) -> None:
        while self._compute_step():
            continue
        self.state.reset_state(self.convert_state_dict_to_array())
        return

    '''
    Take on step in computation. Only do one matrix multiplication between a gate and 
    the quantum state.
    Return value: Return True if the function caclulate a gate. 
    If it does nothing, return False
    '''

    def _compute_step(self) -> bool:
        if self.calc_step == len(self.calc_sequence):
            return False
        if self.Debug:
            print(f"Step :{self.calc_step}")
        (gate, qubit_indices, index) = self.calc_sequence[self.calc_step][0]
        if isinstance(gate, PauliX):
            new_state = {}
            if isinstance(qubit_indices, tuple):
                qubit_indices = qubit_indices[0]
            qubit_index = int(qubit_indices)
            for stateint in self._statedict.keys():
                new_state[self.flip_bit(qubit_index, stateint)] = self._statedict[stateint]
            self._statedict = new_state
        elif isinstance(gate, Hadamard):
            '''
            H=1/\sqrt{2}*(X+Z)
            First, we generate the state for X operation
            Then we generate the state for Z operation
            Finally, we merge the result together
            '''
            new_state = {}
            if isinstance(qubit_indices, tuple):
                qubit_indices = qubit_indices[0]

            qubit_index = int(qubit_indices)
            for stateint in self._statedict.keys():
                new_state[self.flip_bit(qubit_index, stateint)] = 1 / np.sqrt(2) * self._statedict[stateint]

            for stateint in self._statedict.keys():
                if self.bit_is_1(qubit_index, stateint):
                    self._statedict[stateint] = -1 / np.sqrt(2) * self._statedict[stateint]
                else:
                    self._statedict[stateint] = 1 / np.sqrt(2) * self._statedict[stateint]
                if stateint in new_state.keys():
                    self._statedict[stateint] = self._statedict[stateint] + new_state.pop(stateint)
            '''
            Finally, merge two dictionary
            '''
            self._statedict = self._statedict | new_state
        elif isinstance(gate, CNOT):
            control_index = qubit_indices[0]
            target_index = qubit_indices[1]
            newstate = {}
            for stateint in self._statedict.keys():
                if self.bit_is_1(control_index, stateint):
                    newstateint = self.flip_bit(target_index, stateint)
                    newstate[newstateint]=self._statedict[stateint]
                else:
                    newstate[stateint] = self._statedict[stateint]
            self._statedict = newstate
        elif isinstance(gate, TGate):
            if isinstance(qubit_indices, tuple):
                qubit_indices = qubit_indices[0]
            qubit_index = int(qubit_indices)
            phase = np.exp(1j * np.pi / 4)
            if gate.is_dagger():
                phase = np.exp(-1j * np.pi / 4)
            for stateint in self._statedict.keys():
                if self.bit_is_1(qubit_index, stateint):
                    self._statedict[stateint] *= phase
        self._statedict = dict(filter(self.filter_small_value, self._statedict.items()))
        self.calc_step += 1
        return True

    def measure(self, qubit_index: int) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement measure method.")

    def visulize(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement visulize method.")

    def load_qasm(self, qasmStr: str) -> None:
        sentence_list = qasmStr.splitlines()
        version = sentence_list[0]
        qreg = sentence_list[2]
        '''
        First determine the number of qubits by
        sentence such as qreg q[16];
        Use regular expression to extract the number
        '''
        # The regular expression pattern
        # ^ asserts start of a line
        # \d+ matches one or more digits
        # The parentheses ( ) capture the matched digits to a group
        pattern = r'^qreg q\[(\d+)\];'
        # Perform the search
        qubit_num = int(re.match(pattern, qreg).group(1))
        '''
        Re-initialize the quantum state of this circuit
        '''
        self.__init__(qubit_num)
        if version == "OPENQASM 2.0;":
            for i in range(4, len(sentence_list)):
                codelist = sentence_list[i].split(' ')
                instruction = codelist[0]
                registerlist = codelist[1][:-1].split(',')
                regpattern = r'^q\[(\d+)\]'
                match instruction:
                    case 'cx':
                        qubit1 = int(re.match(regpattern, registerlist[0]).group(1))
                        qubit2 = int(re.match(regpattern, registerlist[1]).group(1))
                        qubit_index = [qubit1, qubit2]
                        self.add_gate(Gate.CNOT(), qubit_index)
                    case 'h':
                        qubit_index = int(re.match(regpattern, registerlist[0]).group(1))
                        self.add_gate(Gate.Hadamard(), qubit_index)
                    case 't':
                        qubit_index = int(re.match(regpattern, registerlist[0]).group(1))
                        self.add_gate(Gate.TGate(), qubit_index)
                    case 'tdg':
                        qubit_index = int(re.match(regpattern, registerlist[0]).group(1))
                        gate = Gate.TGate()
                        gate.dagger()
                        self.add_gate(gate, qubit_index)
                    case 'x':
                        qubit_index = int(re.match(regpattern, registerlist[0]).group(1))
                        self.add_gate(Gate.PauliX(), qubit_index)
        return

    def to_qasm(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement to_qasm method.")

    def print_state(self) -> None:
        self.state.show_state()

    def state_vector(self) -> np.ndarray:
        return self.state.state_vector
