import numpy as np

import Gate
import Parameter
from Gate import *
from Gate.Gate import QuantumGate
from State import QuantumState
from typing import List, Union
import re


class QuantumCircuit:
    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = num_qubits

    def add_gate(self, gate: QuantumGate, qubit_indices: list[int]) -> None:
        raise NotImplementedError("Subclasses must implement add_gate method.")

    def clear_all(self) -> None:
        raise NotImplementedError("clear_all must implement clear_all method.")

    def clear_state(self) -> None:
        raise NotImplementedError("clear_all must implement clear_state method.")

    '''
    Implement the computation process of the whole circuit.
    The state should be in the final state after evolving after the compute method
    is called
    '''

    def compute(self) -> NotImplementedError:
        return NotImplementedError("Subclasses must implement compute method.")

    '''
    Measure a subset of, after which the state is still a quantum state, 
    But the state of the measured qubits has collapsed 
    '''

    def measure(self, qubit_indices: List) -> NotImplementedError:
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
        Store the computed matrix for debugging
        Can remove in the future to save space
        '''
        self._matrix = np.identity(1 << num_qubits, Parameter.qtype)
        self.store_matrix = False

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

    def clear_all(self) -> None:
        self.__init__(self.num_qubits)

    def clear_state(self) -> None:
        self.state = QuantumState(qubit_number=self.num_qubits)

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
    Check weather a expanded matrix is still unitary
    '''

    def check_unitary(self, matrix: np.ndarray, dimension: int):
        I = np.matmul(matrix.conjugate(), matrix)
        eye = np.eye(1 << dimension, dtype=Parameter.qtype)
        return np.allclose(I, eye, atol=0.1)

    '''
    Given a quantum gate on single qubit, expand the whole matrix
    '''

    def expand_kron_single(self, gate: QuantumGate, qubit_indices: List) -> np.ndarray:
        '''
        All hadamard gate.
        Assume that there are n qubits,
        Ignore the 1/sqrt{2^n} factor, H^n[i][j]=-1 only when i,j are both odd, otherwise H^n[i][j]=1
        '''
        if isinstance(gate, AllHadamard):
            N = 2 ** gate.num_qubits
            matrix = np.ones((N, N), dtype=Parameter.qtype)
            for i in range(0, N):
                for j in range(0, N):
                    ilist = self.bit_list(self.num_qubits, i)
                    jlist = self.bit_list(self.num_qubits, j)
                    ijlist = [ilist[q] * jlist[q] for q in range(self.num_qubits)]
                    if sum(ijlist) % 2 == 1:
                        matrix[i][j] = -1
            matrix = matrix / np.sqrt(N)
            if self.Debug:
                if not self.check_unitary(matrix, self.num_qubits):
                    raise ValueError("Matrix not unitary")
            return matrix
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
            qubit_index = qubit_indices[0]
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
            if self.Debug:
                if not self.check_unitary(matrix, self.num_qubits):
                    raise ValueError("Matrix not unitary")
            return matrix
        elif gate.num_qubits >= 3:
            '''
            The construction process is different between normal gate with multiControlX and multiControlZ
            '''
            if not isinstance(gate, MultiControlX) and not isinstance(gate, MultiControlZ):
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
            elif isinstance(gate, MultiControlX):
                '''
                When the gate is a multiControlled X
                We should check weather the controlled condition is satisfied 
                before we set the matrix element to 1
                For example, when the act_condition=[0,1], and the act_index=2
                |010> will change to |011>,  
                |011> will change to |010> 
                '''
                control_indices = qubit_indices[0]
                act_index = qubit_indices[1]
                act_condition = gate.act_condition
                matrix = np.identity(1 << self.num_qubits, dtype=Parameter.qtype)
                '''
                All 0,1 element must be the same except the control qubit and the act qubit
                '''
                maskint = (1 << (self.num_qubits - act_index - 1))
                maskint = ~maskint
                maskint = maskint % (1 << self.num_qubits)
                for column in range(0, 1 << self.num_qubits):
                    for row in range(0, 1 << self.num_qubits):
                        if (row & maskint) != (column & maskint):
                            matrix[row][column] = 0
                            continue
                        checkControl = True
                        '''
                        Check that whether the bit status of all the control qubits match the act_condition
                        '''
                        for control_index in range(len(control_indices)):
                            if self.bitstatus(control_indices[control_index], column) != act_condition[control_index]:
                                matrix[row][column] = 0
                                checkControl = False
                                break
                        if not checkControl:
                            if self.bitstatus(act_index, column) != (self.bitstatus(act_index, row)):
                                matrix[row][column] = 0
                                continue
                            matrix[row][column] = 1
                            continue
                        '''
                        When the control condition holds, check whether the matrix flips the act qubit
                        '''
                        if self.bitstatus(act_index, column) != ((self.bitstatus(act_index, row) + 1) % 2):
                            matrix[row][column] = 0
                            continue
                        matrix[row][column] = 1
            elif isinstance(gate, MultiControlZ):
                '''
                When the gate is a multiControlled Z
                We should check weather the controlled condition is satisfied 
                before we set the matrix element to 1
                For example, when the act_condition=[0,1], and the act_index=2
                |011> will change to -|011>,  
                |010> will not change
                '''
                control_indices = qubit_indices[0]
                act_index = qubit_indices[1]
                act_condition = gate.act_condition
                matrix = np.identity(1 << self.num_qubits, dtype=Parameter.qtype)
                for column in range(0, 1 << self.num_qubits):
                    checkControl = True
                    '''
                        Check that whether the bit status of all the control qubits match the act_condition
                        '''
                    for control_index in range(len(control_indices)):
                        if self.bitstatus(control_indices[control_index], column) != act_condition[control_index]:
                            matrix[column][column] = 1
                            checkControl = False
                            break
                    if not checkControl:
                        continue
                    '''
                    When the control condition holds, check whether the element is 1
                    '''
                    if self.bitstatus(act_index, column) != 1:
                        matrix[column][column] = 1
                        continue
                    matrix[column][column] = -1
        if self.Debug:
            if not self.check_unitary(matrix, self.num_qubits):
                raise ValueError("Matrix not unitary")
        return matrix

    '''
    Give a list of gates in the same column/layer, expand them to the whole matrix
    For example, when the list is [H,H,H], it should return the same matrix as all hadamard gate.
    '''

    def expand_kron_multi(self, gatelist: List) -> np.ndarray:
        return np.array([0, 0])

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
        if self.calc_step == len(self.calc_sequence):
            return False
        if self.Debug:
            print(f"Step :{self.calc_step}")
        '''
        First get the whole matrix after doing kron product
        '''
        if len(self.calc_sequence[self.calc_step]) == 1:
            (gate, qubit_indices, index) = self.calc_sequence[self.calc_step][0]
            matrix = self.expand_kron_single(gate, qubit_indices)
        else:
            matrix = self.expand_kron_multi(self.calc_sequence[self.calc_step])
        if self.Debug:
            print(f"The {self.calc_step} step of calculation, the matrix is \n {matrix}")
        self.state.reset_state(np.matmul(matrix, self.state.state_vector))
        if self.store_matrix:
            self._matrix = np.matmul(matrix, self._matrix)
        self.calc_step += 1
        return True

    '''
    Check weather the qubit status represented by integer pos match the qubit_status list, given the qubit_indices
    For example, when pos=2=010, the qubit indices=[1,2], and the qubit_status=[0,1], the answer should be true
    because the status of the last qubits are indeed 01
    '''

    def match_index(self, pos: int, qubit_indices: List, qubit_status: List):
        return

    '''
    Calculate the bitlist of pos
    For example, when pos=010=2, if will return [0,1,0]
    '''

    def bit_list(self, num_qubits: int, pos: int):
        bitlist = []
        k = pos
        for i in range(0, num_qubits):
            bit = k % 2
            bitlist.insert(0, int(bit))
            k = (k >> 1)
        return bitlist

    def state_vector(self) -> np.ndarray:
        return self.state.state_vector

    '''
    Measurement operation, return the probability list of measurement 
    For example, if the state after computation is
    |000>/2**0.5+|111>/2**0.5, and the qubit indices is set to [0,2], then the probabilities 
    of measuring |00>,|01>,|10>,|11> are 0.5,0,0,0.5. (Doing inner-product)
    After we randomly choose a result, say 00 is the outcome, the state collapses to |000>
    The function return a list of 0,1. For example, when the measurement result is |00>, it will return [0,0].
    '''

    def measure(self, qubit_indices: List) -> List:
        '''
        First make sure the computation is done
        '''
        if self.calc_step != self.num_qubits:
            self.compute()
        l = len(qubit_indices)
        '''
        The final probability of all possible measurement result
        in the subsystem
        '''
        problist = [0] * (1 << l)
        state_vector = self.state_vector()
        # self.state.show_state_dirac()
        for i in range(0, 1 << self.num_qubits):
            qubit_status_list = self.bit_list(self.num_qubits, i)
            sub_status_list = [qubit_status_list[index] for index in qubit_indices]
            pos = 0
            for j in range(0, l):
                pos = pos + (sub_status_list[j] << (l - j - 1))
            problist[pos] += (state_vector[i] * np.conjugate(state_vector[i]))
        '''
        Randomly select a result based on the problist
        '''
        problist = [np.real(x / sum(problist)) for x in problist]
        result = np.random.choice(list(range(0, 1 << l)), 1, p=problist)
        result = self.bit_list(l, result)
        '''
        Collapse the final state after this measurement
        '''
        normalization_factor = 0
        for i in range(0, 1 << self.num_qubits):
            qubit_status_list = self.bit_list(self.num_qubits, i)
            sub_status_list = [qubit_status_list[index] for index in qubit_indices]
            if sub_status_list != result:
                state_vector[i] = 0
            else:
                normalization_factor += (state_vector[i] * np.conjugate(state_vector[i]))
        state_vector = state_vector / (np.sqrt(normalization_factor))
        self.state.reset_state(state_vector)
        return result

    '''
    Measureall qubit. 
    We require users to set the expected qubit bitstring outcome of this measurement, and 
    we will return the probability to them with regard to the given bitstring.
    '''

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

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix


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
