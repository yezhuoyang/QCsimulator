import sys

sys.path.append('..')
import Circuit
import Gate
'''
C = Circuit.NumpyCircuit(2)
C.Debug = True
C.add_gate(Gate.CNOT(), [1, 0])
C.compute()
C.print_state()

D = Circuit.NumpyCircuit(4)
D.Debug = True
D.add_gate(Gate.Hadamard(), 0)
D.add_gate(Gate.Hadamard(), 1)
D.add_gate(Gate.Hadamard(), 2)
D.add_gate(Gate.Hadamard(), 3)
D.compute()
D.print_state()
'''


def test_measure():
    return


def test_measure_all():
    return







C = Circuit.NumpyCircuit(3)
C.Debug = True
C.add_gate(Gate.Toffoli(), [0,1,2])
C.compute()
C.print_state()