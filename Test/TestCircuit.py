import sys

sys.path.append('..')
import Circuit
import Gate

C = Circuit.NumpyCircuit(1)
C.Debug = True
C.add_gate(Gate.Hadamard(), 0)
C.add_gate(Gate.Hadamard(), 0)
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



