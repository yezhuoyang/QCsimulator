import qiskit
from qiskit import execute
import functools
from qiskit_aer import AerSimulator as Aer
from qiskit.circuit.library.standard_gates import HGate, CXGate, TGate, TdgGate, CCXGate
from Gate.Gate import Toffoli
import qiskit.quantum_info as qi
import numpy as np

'''
Optimization of Toffoli gate using two 
qubit and single qubit gates
'''



toffoli_matrix = Toffoli().matrix()


def correct_matrix():
    circuit = qiskit.QuantumCircuit(3)
    circuit.append(CCXGate(),[0,1,2])
    op = qi.Operator(circuit)
    return np.array(op)


def get_matrix(circuit: qiskit.QuantumCircuit):
    op = qi.Operator(circuit)
    return np.array(op)


'''
Statndard implementation of exact toffoli gate
We use six CNOT gates, 2 hadamard gates, 4 T gates and 3 Tdagger gates.
'''


def standard_solution():
    circuit = qiskit.QuantumCircuit(3)
    circuit.append(HGate(), [2])
    circuit.append(CXGate(), [1, 2])
    circuit.append(TdgGate(), [2])
    circuit.append(CXGate(), [0, 2])
    circuit.append(TGate(), [2])
    circuit.append(CXGate(), [1, 2])
    circuit.append(TdgGate(), [2])
    circuit.append(CXGate(), [0, 2])
    circuit.append(TGate(), [1])
    circuit.append(TGate(), [2])
    circuit.append(CXGate(), [0, 1])
    circuit.append(TGate(), [0])
    circuit.append(TdgGate(), [1])
    circuit.append(CXGate(), [0, 1])
    circuit.append(HGate(), [2])
    return circuit


'''
Five two qubit gate 
'''


def five_two_qubit_solution():
    return


'''
Four two qubit gate solution
Only allows two qubit gates between AB and BC 
'''


def five_two_qubit_solution():
    return


def hilbert_schmidt_distance(matrix1: np.ndarray, matrix2: np.ndarray):
    return np.trace((matrix1 - matrix2) ** 2)


'''
Optimize and output the bset solution
'''


def optimize():
    return


if __name__ == "__main__":


    print(hilbert_schmidt_distance(correct_matrix(), get_matrix(standard_solution())))
