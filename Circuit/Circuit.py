import numpy as np

import Gate
import Parameter
from Gate import *
from Gate.Gate import QuantumGate
from State import QuantumState
from typing import List, Union, Any
import re


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

    def load_qasm(self, qasmStr: str) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement load_qasm method.")

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
        self.Debug = False

    def print_state(self) -> None:
        self.state.show_state()

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
            if qubit_indices < 0 or qubit_indices >= self.num_qubits:
                raise ValueError(f"qubit_indices {qubit_indices} out of range!")
            '''
            The step is to make sure it can be converted to tuple in the fowllowing steps.
            '''
            qubit_indices=[qubit_indices]
        elif isinstance(qubit_indices, List):
            if gate.num_qubits != len(qubit_indices):
                raise ValueError("The length of the qubit_indices list has to match with the qubit number of the gate!")
            for index in qubit_indices:
                if index < 0 or index >= self.num_qubits:
                    raise ValueError(f"{index} in qubit_indices {qubit_indices} out of range!")
            '''Make sure that the qubit indices of single qubit gate be just an integer'''
            if gate.num_qubits == 1:
                qubit_indices = qubit_indices[0]
        self.gate_list.append((gate, tuple(qubit_indices), self.gate_num))
        self.calc_dict[(gate, tuple(qubit_indices), self.gate_num)] = False
        self.gate_num += 1

    '''
    Given an integer state_int, return the bit status of qubit qubit_index
    regarding to the bit in the binary representation of state_int.
    For example, when state_int=3, the circuit has 4 qubit, and the qubit_index=1
    First we transform 3 into binary form 101, and we found that the 1th qubit(count from
    left to right) has status 0, so the function will return 0. 
    TODO: How to optimize the bitstatus and matrix construction function?
    '''

    def bitstatus(self, qubit_index: int, state_int: int):
        '''
        Before calculation, check whether the state_int and
        qubit_index is valid.
        For performance in the future, this code will be muted.
        '''
        if qubit_index >= self.num_qubits or state_int < 0 or state_int >= (1 << self.num_qubits):
            raise ValueError("qubit_index or state_int out of range!")
        return (state_int >> (self.num_qubits - qubit_index - 1)) & 1

    '''
    TODO:Calculate the matrix form of a gate after kron product
    This part is the most difficult because for a 2-qubit gates,
    we have to swap two index together before we can do simple kroneck product
    '''

    def calc_kron(self, gateindex: int) -> np.ndarray:
        (gate, qubit_indices, index) = self.gate_list[gateindex]
        '''
        If the circuit has only one qubit, just return the matrix of single
        gate
        '''
        if self.num_qubits == 1:
            return gate.matrix()
        I = np.array([[1, 0], [0, 1]], dtype=Parameter.qtype)
        '''
        Matrix form for single qubit gate, we can simply use np.kron to
        generate the final matrix
        '''
        matrix = I
        if gate.num_qubits == 1:
            qubit_index=qubit_indices[0]
            if qubit_index == 0:
                matrix = gate.matrix()
            else:
                for i in range(0, qubit_index - 1):
                    matrix = np.kron(matrix, I)
                matrix = np.kron(matrix, gate.matrix())
            for i in range(qubit_index, self.num_qubits - 1):
                matrix = np.kron(matrix, I)
        elif gate.num_qubits == 2:
            gatematrix = gate.matrix()
            matrix = np.identity(1 << self.num_qubits, dtype=Parameter.qtype)
            for column in range(0, 1 << self.num_qubits):
                for row in range(0, 1 << self.num_qubits):
                    print(column,row)
                    i = qubit_indices[0]
                    j = qubit_indices[1]
                    '''
                    If the bit-status between column and row in any of the 
                    position except i,j are different, Matrix[row][column] must 
                    be 0. This can be done by using == operator after we mask i,j th
                    qubit 
                    '''
                    maskint = ~((1 << (self.num_qubits - i - 1)) + (1 << (self.num_qubits - j - 1)))
                    if (row & maskint) != (column & maskint):
                        matrix[row][column] = 0
                        continue
                    '''
                    When all bit states except qubit i,j are the same. We 
                    should further determine which element from the gate matrix
                    should we put here.
                    The element to be put in matrix[row][column] should be:
                    <ki_l kj_l | GateMatrix |ki_r kj_r>
                    Which is simply
                                  GateMatrix_{ki_l kj_l, ki_r kj_r}
                    '''
                    ki_l = self.bitstatus(i, row)
                    kj_l = self.bitstatus(j, row)
                    gaterowindex = (ki_l << 1) + kj_l
                    ki_r = self.bitstatus(i, column)
                    kj_r = self.bitstatus(j, column)
                    gatecolindex = (ki_r << 1) + kj_r
                    if self.Debug:
                        print(f"ki_l:{ki_l} kj_l:{kj_l}   ki_r:{ki_r}  kj_r:{kj_r}")
                        print(gaterowindex, gatecolindex)
                    matrix[row][column] = gatematrix[gaterowindex][gatecolindex]
            '''
            TODO: Two or three qubit gates
            '''
            return matrix
        elif gate.num_qubits == 3:
            gatematrix = gate.matrix()
            matrix = np.identity(1 << self.num_qubits, dtype=Parameter.qtype)
            for column in range(0, 1 << self.num_qubits):
                for row in range(0, 1 << self.num_qubits):
                    i = qubit_indices[0]
                    j = qubit_indices[1]
                    q = qubit_indices[2]
                    '''
                    Just the same as two qubit gate case 
                    If the bit-status between column and row in any of the 
                    position except i,j,q are different, Matrix[row][column] must 
                    be 0. This can be done by using == operator after we mask i,j,q th
                    qubit 
                    '''
                    maskint = ~((1 << (self.num_qubits - i - 1)) + (1 << (self.num_qubits - j - 1)) + (
                            1 << (self.num_qubits - q - 1)))
                    if (row & maskint) != (column & maskint):
                        matrix[row][column] = 0
                        continue
                    '''
                    Just the same as two qubit gate
                    When all bit states except qubit i,j,q are the same. We 
                    should further determine which element from the gate matrix
                    should we put here.
                    The element to be put in matrix[row][column] should be:
                    <ki_l kj_l kq_l | GateMatrix |ki_r kj_r kq_r>
                    Which is simply
                                  GateMatrix_{ki_l kj_l kq_l, ki_r kj_r kq_r}
                    '''
                    ki_l = self.bitstatus(i, row)
                    kj_l = self.bitstatus(j, row)
                    kq_l = self.bitstatus(q, row)
                    gaterowindex = (ki_l << 2) + (kj_l << 1) + kq_l
                    ki_r = self.bitstatus(i, column)
                    kj_r = self.bitstatus(j, column)
                    kq_r = self.bitstatus(q, column)
                    gatecolindex = (ki_r << 2) + (kj_r << 1) + kq_r
                    if self.Debug and gatematrix[gaterowindex][gatecolindex] != 0:
                        print(f"ki_l:{ki_l} kj_l:{kj_l} kq_l:{kq_l}  ki_r:{ki_r}  kj_r:{kj_r} kq_r:{kq_r}")
                        print(f"Element {row},{column} is set to Gate{gaterowindex}, {gatecolindex}")
                    matrix[row][column] = gatematrix[gaterowindex][gatecolindex]
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
        print(f"Step :{self.calc_step}")
        '''
        Avoid repeated computation
        '''
        if self.calc_dict[self.gate_list[self.calc_step]]:
            return False
        '''
        First get the whole matrix after doing kron product
        '''
        matrix = self.calc_kron(self.calc_step)
        if self.Debug:
            print(f"The {self.calc_step} step of calculation, the matrix is \n {matrix}")
        self.state.reset_state(np.matmul(matrix, self.state.state_vector))
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

    def measureAll(self, statestr: str) -> np.complex128:
        '''
        First make sure the computation is done
        '''
        if self.calc_step != self.num_qubits:
            self.compute()
        if len(statestr) != self.num_qubits:
            raise ValueError("Qubit number does not match in the measurement!")
        return np.complex128(abs(self.state.state_vector[int(statestr, 2)]) ** 2)

    def visulize(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement visulize method.")

    def load_qasm(self, qasmStr) -> None:
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

    def to_qasm(self) -> str:
        return "Not"

    def state_vector(self) -> np.ndarray:
        return self.state.state_vector
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

    def load_qasm(self, qasmStr: str) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement load_qasm method.")

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

    def load_qasm(self, qasmStr: str) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement load_qasm method.")

    def to_qasm(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement to_qasm method.")
