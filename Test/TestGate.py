import sys
import unittest

import numpy as np

import Gate
from Circuit import NumpyCircuit
import Parameter
from Gate import QuantumGate

sys.path.append('..')

'''
Check that all gates are unitary
'''

single_gate_list = [Gate.TGate(), Gate.PauliX(), Gate.PauliY(), Gate.PauliZ(), Gate.Hadamard(), Gate.Phase()]
two_gate_list = [Gate.CNOT(), Gate.ControlledZ(), Gate.CPhase(), Gate.Swap()]
three_gate_list = [Gate.Toffoli(), Gate.Fredkin()]


def check_unitary(gate: QuantumGate):
    qubit_number = gate.num_qubits
    identity = np.identity(1 << qubit_number, dtype=Parameter.qtype)
    gate_matrix = gate.matrix()
    gate.dagger()
    gate_dagger_matrix = gate.matrix()
    gateid = np.matmul(gate_matrix, gate_dagger_matrix)
    return np.allclose(identity, gateid, Parameter.unitary_test_tol, Parameter.unitary_test_tol)


'''
Check Toffoli identity
'''


def check_toffoli():
    testcircuit = NumpyCircuit(3)
    testcircuit.store_matrix = True
    testcircuit.add_gate(Gate.Hadamard(), 2)
    testcircuit.add_gate(Gate.CNOT(), [1, 2])
    Tdag = Gate.TGate()
    Tdag.dagger()
    testcircuit.add_gate(Tdag, 2)
    testcircuit.add_gate(Gate.CNOT(), [0, 2])
    testcircuit.add_gate(Gate.TGate(), 2)
    testcircuit.add_gate(Gate.CNOT(), [1, 2])
    testcircuit.add_gate(Tdag, 2)
    testcircuit.add_gate(Gate.CNOT(), [0, 2])
    testcircuit.add_gate(Gate.TGate(), 1)
    testcircuit.add_gate(Gate.TGate(), 2)
    testcircuit.add_gate(Gate.Hadamard(), 2)
    testcircuit.add_gate(Gate.CNOT(), [0, 1])
    testcircuit.add_gate(Gate.TGate(), 0)
    testcircuit.add_gate(Tdag, 1)
    testcircuit.add_gate(Gate.CNOT(), [0, 1])
    testcircuit.compute()
    toffolimatrix = Gate.Toffoli().matrix()
    return np.allclose(testcircuit.matrix, toffolimatrix, Parameter.unitary_test_tol, Parameter.unitary_test_tol)


class TestGate(unittest.TestCase):
    def test_single_unitary(self):
        for gate in single_gate_list:
            self.assertEqual(check_unitary(gate), True,
                             msg=f"{gate} not unitary! The matrix is {np.matmul(gate.matrix(), gate.matrix().conjugate())}")

    def test_two_unitary(self):
        for gate in two_gate_list:
            self.assertEqual(check_unitary(gate), True, msg=f"{gate} not unitary! The matrix is {gate.matrix()}")

    def test_three_unitary(self):
        for gate in three_gate_list:
            self.assertEqual(check_unitary(gate), True, msg=f"{gate} not unitary! The matrix is {gate.matrix()}")

    def check_toffoli(self):
        self.assertEqual(check_toffoli(), True, msg=f"Toffoli check fail!")


def main():
    unittest.main()


if __name__ == "__main__":
    main()
