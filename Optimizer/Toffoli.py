import qiskit
from qiskit import execute
import functools
from qiskit_aer import AerSimulator as Aer
from qiskit.circuit.library.standard_gates import HGate, CXGate, TGate, TdgGate, CCXGate, RZGate, RXGate, RYGate, \
    CSdgGate, CSGate, U1Gate
from Gate.Gate import Toffoli
import qiskit.quantum_info as qi
import numpy as np
from scipy.optimize import minimize

'''
Optimization of Toffoli gate using two 
qubit and single qubit gates
'''

toffoli_matrix = Toffoli().matrix()


def correct_matrix():
    circuit = qiskit.QuantumCircuit(3)
    circuit.append(CCXGate(), [0, 1, 2])
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


def five_two_qubit_solution_1(params):
    circuit = qiskit.QuantumCircuit(3)

    circuit.append(HGate(), [2])
    circuit.append(CXGate(), [0, 1])

    circuit.append(RXGate(params[0]), [0])
    circuit.append(RZGate(params[1]), [1])

    circuit.append(CXGate(), [1, 2])

    circuit.append(RZGate(params[2]), [1])
    circuit.append(RZGate(params[3]), [2])

    circuit.append(CXGate(), [0, 1])

    circuit.append(RYGate(params[4]), [0])
    circuit.append(RZGate(params[5]), [1])

    circuit.append(CXGate(), [1, 2])

    circuit.append(RXGate(params[6]), [1])
    circuit.append(RZGate(params[7]), [2])

    circuit.append(CXGate(), [0, 1])
    circuit.append(HGate(), [2])
    return circuit





'''
Five two qubit gate 
'''


def five_two_qubit_solution_2(params):
    circuit = qiskit.QuantumCircuit(3)
    circuit.append(HGate(), [2])

    circuit.append(CSGate(), [1, 2])
    circuit.append(U1Gate(params[0]), [0])
    circuit.append(CXGate(), [0, 1])

    circuit.append(U1Gate(params[1]), [2])
    circuit.append(CSdgGate(), [1, 2])
    circuit.append(U1Gate(params[2]), [2])

    circuit.append(CXGate(), [0, 1])

    circuit.append(CSGate(), [1, 2])

    circuit.append(U1Gate(params[3]), [2])

    circuit.append(HGate(), [2])
    return circuit


'''
Four two qubit gate solution
Only allows two qubit gates between AB and BC 
'''


def four_two_qubit_solution(params):
    circuit = qiskit.QuantumCircuit(3)
    circuit.append(HGate(), [2])

    circuit.append(CSGate(), [1, 2])
    circuit.append(CXGate(), [0, 1])
    circuit.append(CSdgGate(), [1, 2])

    circuit.append(CXGate(), [0, 1])

    circuit.append(U1Gate(params[0]), [0])
    circuit.append(U1Gate(params[1]), [2])

    circuit.append(HGate(), [2])
    return circuit


def hilbert_schmidt_distance(V: np.ndarray, U: np.ndarray):
    L = V.shape[0]
    Vdag = V.transpose()
    Vdag = Vdag.conjugate()
    return np.sqrt(1 - np.abs(np.abs(np.trace(np.matmul(Vdag, U))) ** 2) / (L ** 2))


def norm_2_distance(matrix1: np.ndarray, matrix2: np.ndarray):
    return np.abs(np.trace((matrix1 - matrix2) ** 2))


def loss(params):
    dis = hilbert_schmidt_distance(get_matrix(five_two_qubit_solution_2(params)), correct_matrix())
    print(dis)
    return dis


'''
Optimize and output the bset solution
'''


def optimize():
    params = np.random.uniform(low=0, high=2 * np.pi, size=4)
    # params = [np.pi / 4, np.pi / 4]
    bounds = [(0, 2 * np.pi) for _ in range(4)]
    res = minimize(loss, params, method="COBYLA", bounds=bounds)
    print(res.x)
    return res.x


if __name__ == "__main__":
    params = optimize()
    print(get_matrix(four_two_qubit_solution(params)))

    print(correct_matrix())

    print(get_matrix(standard_solution()))
