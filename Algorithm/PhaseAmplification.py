import qiskit
import functools
from qiskit_aer import AerSimulator
from Algorithm import QuantumAlgorithm
import Gate
import Parameter
import Circuit
from typing import List, Union, Any
from util import convert_int_to_list, convert_list_to_int


class PhaseAmplification(QuantumAlgorithm):

    def __init__(self, num_qubits: int) -> None:
        super().__init__(num_qubits)
        self._computed = False
        self._constructed = False
        self._circuit = qiskit.QuantumCircuit(num_qubits, num_qubits - 1)
        self._simulator = AerSimulator()
        self._num_qubits = num_qubits
        '''
        How many time we may need to call the phaseampli operator,
        initially set to be 1
        '''
        self._phaseampli_step = 0
        self._max_step = 10
        self._database = []
        self._solution = -1
        self._baselinegates = []
        self._reverse_baselinegates = []
        self._succeed = False

    def construct_circuit(self) -> None:
        if self._constructed:
            return
        # self.circuit.h(list(range(0, self.num_qubits)))
        '''
        Construct phaseampli circuit many times
        '''
        self.circuit.x(self.num_qubits - 1)
        self.construct_baseline()
        for i in range(0, self._phaseampli_step):
            self.construct_phaseampli_op()
        self._circuit.measure(list(range(0, self.num_qubits - 1)), list(range(0, self.num_qubits - 1)))
        self._constructed = True
        return

    @property
    def circuit(self):
        return self._circuit

    @property
    def phaseampli_step(self):
        return self._phaseampli_step

    @phaseampli_step.setter
    def phaseampli_step(self, step: int):
        self._phaseampli_step = step
        self.clear_circuit()

    def clear_circuit(self):
        self._circuit.clear()
        self._constructed = False

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
                        self._circuit.cz(0, 1)
                    elif i == 0:
                        self._circuit.x(0)
                        self._circuit.cz(0, 1)
                        self._circuit.x(0)
                else:
                    cond_lis = convert_int_to_list(self.num_qubits - 1, i)
                    for i in range(0, self.num_qubits - 1):
                        if cond_lis[i] == 0:
                            self._circuit.x(i)
                    # There is not mcz in qiskit, we have to use the identity HXH=Z
                    self._circuit.h(self.num_qubits - 1)
                    self._circuit.mcx(control_qubits=list(range(0, self.num_qubits - 1)),
                                      target_qubit=self.num_qubits - 1)
                    self._circuit.h(self.num_qubits - 1)
                    for i in range(0, self.num_qubits - 1):
                        if cond_lis[i] == 0:
                            self._circuit.x(i)
        return

    '''
    zo add a (-1) phase to all states |00...00>
    '''

    def construct_zo(self):
        if self.num_qubits == 2:
            self._circuit.x(0)
            self._circuit.cz(0, 1)
            self._circuit.x(0)
        else:
            for i in range(0, self.num_qubits - 1):
                self._circuit.x(i)
            self._circuit.h(self.num_qubits - 1)
            self._circuit.mcx(control_qubits=list(range(0, self.num_qubits - 1)),
                              target_qubit=self.num_qubits - 1)
            self._circuit.h(self.num_qubits - 1)
            for i in range(0, self.num_qubits - 1):
                self._circuit.x(i)
        return

    def construct_phaseampli_op(self):
        qreglist = list(range(0, self.num_qubits))
        self.construct_zf()
        self._circuit.barrier(qreglist)
        self.construct_baseline(dagger=True)
        self._circuit.barrier(qreglist)
        self.construct_zo()
        self._circuit.barrier(qreglist)
        self.construct_baseline()
        self._circuit.barrier(qreglist)
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

    def set_baseline(self, gatelist: List) -> None:
        self._baselinegates = gatelist
        self._reverse_baselinegates = list(reversed(gatelist))

    def construct_baseline(self, dagger=False) -> None:
        if not dagger:
            for (gate, indices) in self._baselinegates:
                self._circuit.append(gate, indices)
        else:
            for (gate, indices) in self._reverse_baselinegates:
                self._circuit.append(gate.inverse(), indices)

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
        compiled_circuit = qiskit.transpile(self._circuit, self._simulator)
        # Execute the circuit on the aer simulator
        job = self._simulator.run(compiled_circuit, shots=1)

        # Grab results from the job
        result = job.result()
        # Returns counts
        counts = result.get_counts(compiled_circuit)
        print(counts)
        result = list(counts.keys())[0]

        self._computed = False
        result = int(result[::-1], 2)
        if self._database[result] == 1:
            self._computed = True
            self._succeed = True
            self._solution = result
            print(f"Found a solution {result}, Phaseampli Step={self._phaseampli_step}")
        else:
            print("Algorithm failed!")

        return

    '''
    Calculate the succcess rate before and after phase amplification
    So we can compare the two results and understand the effect of phase amplification.
    '''

    def calc_success_rate(self, shotnum: int):
        self.construct_circuit()
        compiled_circuit = qiskit.transpile(self._circuit, self._simulator)
        # Execute the circuit on the aer simulator
        job = self._simulator.run(compiled_circuit, shots=shotnum)
        # Grab results from the job
        result = job.result()
        # Returns counts
        counts = result.get_counts(compiled_circuit)
        print(counts)
        succcess_num = 0

        for key in counts.keys():
            str_list = list(key)
            str_list = [int(x) for x in str_list]
            keyvalue = convert_list_to_int(self.num_qubits - 1, str_list)
            # print("Key: %s   keyvalue: %d   count:%d" % (key, keyvalue, counts[key]))
            if self._database[keyvalue] == 1:
                succcess_num += counts[key]
        return succcess_num / shotnum

    @property
    def solution(self) -> int:
        return self._solution

    def set_simulator(self, simulator):
        self._simulator = simulator


from qiskit.circuit.library.standard_gates import HGate, TGate, XGate, SdgGate, CSdgGate
from qiskit.circuit.library import UnitaryGate
import numpy as np

if __name__ == "__main__":
    B = PhaseAmplification(3)
    B.set_input([0, 0, 0, 1])
    matrix = np.array([[1, -1j], [-1j, 1]] / np.sqrt(2))
    Rgate = UnitaryGate(matrix, label="R")
    Csdg_1 = SdgGate().control(1)
    Csdg_2 = SdgGate().control(2)
    GateList = [(HGate(), [0]), (HGate(), [1]), (Csdg_1, [0, 2]), (Csdg_2, [0, 1, 2]), (Rgate, [0]),
                (Rgate, [1])]
    B.set_baseline(GateList)

    for step in range(0, 3):
        B.phaseampli_step = step
        rate = B.calc_success_rate(200)
        print("Step: %d,  Rate: %.2f  " % (step, rate))
