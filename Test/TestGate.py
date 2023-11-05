
import sys

import Gate

sys.path.append('..')
import Circuit

C = Circuit.NumpyCircuit(1)
C.Debug = True
C.add_gate(Gate.AllHadamard(1), 0)
C.compute()
C.print_state()
