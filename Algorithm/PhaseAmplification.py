import qiskit
import functools
from qiskit_aer import AerSimulator
from Algorithm import QuantumAlgorithm
import Gate
import Parameter
import Circuit
from typing import List, Union, Any
from .util import convert_int_to_list, convert_list_to_int



class PhaseAmplification(QuantumAlgorithm):

    def __init__(self, num_qubits: int) -> None:
        super().__init__(num_qubits)
        self.computed = False
        self.circuit = qiskit.QuantumCircuit(num_qubits, num_qubits - 1)
        self._simulator = AerSimulator()
        self.num_qubits = num_qubits
        '''
        How many time we may need to call the phaseampli operator,
        initially set to be 1
        '''
        self.phaseampli_step = 1
        self.max_step = 10
        self._database = []
        self._solution = -1
        self._baselinegates=[]
        self._succeed = False

    def construct_circuit(self) -> None:
        #self.circuit.h(list(range(0, self.num_qubits)))
        '''
        Construct phaseampli circuit many times
        '''
        for i in range(self.phaseampli_step):
            self.construct_phaseampli_op()
        self.circuit.measure(list(range(0, self.num_qubits - 1)), list(range(0, self.num_qubits - 1)))
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
                        self.circuit.cz(0, 1)
                    elif i == 0:
                        self.circuit.x(0)
                        self.circuit.cz(0, 1)
                        self.circuit.x(0)
                else:
                    cond_lis = convert_int_to_list(self.num_qubits - 1, i)
                    for i in range(0, self.num_qubits - 1):
                        if cond_lis[i] == 0:
                            self.circuit.x(i)
                    # There is not mcz in qiskit, we have to use the identity HXH=Z
                    self.circuit.h(self.num_qubits - 1)
                    self.circuit.mcx(control_qubits=list(range(0, self.num_qubits - 1)),
                                     target_qubit=self.num_qubits - 1)
                    self.circuit.h(self.num_qubits - 1)
                    for i in range(0, self.num_qubits - 1):
                        if cond_lis[i] == 0:
                            self.circuit.x(i)
        return

    '''
    zo add a (-1) phase to all states |00...00>
    '''

    def construct_zo(self):
        if self.num_qubits == 2:
            self.circuit.x(0)
            self.circuit.cz(0, 1)
            self.circuit.x(0)
        else:
            for i in range(0, self.num_qubits - 1):
                self.circuit.x(i)
            self.circuit.h(self.num_qubits - 1)
            self.circuit.mcx(control_qubits=list(range(0, self.num_qubits - 1)),
                             target_qubit=self.num_qubits - 1)
            self.circuit.h(self.num_qubits - 1)
            for i in range(0, self.num_qubits - 1):
                self.circuit.x(i)
        return

    def construct_phaseampli_op(self):
        self.construct_zf()
        self.construct_baseline()
        self.construct_zo()
        self.construct_baseline()
        return


    '''
    Set the baseline operator for Phase amplification
    The input is a list of tuples :(gate,indices)
    For example:
    gatelist=[(Csdg, [0, 1, 2]),(Rgate, [0]),(Rgate, [1])]
    To construct the baseline circuit, we should execute the following commands:
        circuit.append(Csdg, [0, 1, 2])
        circuit.append(Rgate, [0])
        circuit.append(Rgate, [1])
    '''
    def set_baseline(self,gatelist:List)->None:
        self._baselinegates=gatelist



    def construct_baseline(self)->None:
        for (gate,indices) in self._baselinegates:
            self.circuit.append(gate,indices)



    '''
    The input of phaseampli  is the value stored in the 
    unstructured database
    '''

    def set_input(self, database: List) -> None:
        if len(database) != (1 << (self.num_qubits - 1)):
            raise ValueError("The size of the database doesn't match the qubit number")
        self._database = database

    '''
    Complete Phase amplification algorithm, once we find a solution, return 
    Otherwise we add the phaseampli step and do the algorithm again
    '''

    def compute_result(self) -> None:
        # print(self._database)

        self.construct_circuit()
        compiled_circuit = qiskit.transpile(self.circuit, self._simulator)
        # Execute the circuit on the aer simulator
        job = self._simulator.run(compiled_circuit, shots=1)

        # Grab results from the job
        result = job.result()
        # Returns counts
        counts = result.get_counts(compiled_circuit)
        print(counts)
        result = list(counts.keys())[0]

        self.computed = False
        result = int(result[::-1], 2)
        if self._database[result] == 1:
            self.computed = True
            self._succeed = True
            self._solution = result
            print(f"Found a solution {result}, Phaseampli Step={self.phaseampli_step}")
        else:
            print("Algorithm failed!")

        return

    @property
    def solution(self) -> int:
        return self._solution

    def set_simulator(self, simulator):
        self._simulator = simulator