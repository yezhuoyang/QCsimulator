import sys

import Gate

sys.path.append('..')
import Circuit

C = Circuit.NumpyCircuit(1)
C.Debug = True
C.add_gate(Gate.AllHadamard(1), 0)
C.compute()
C.print_state()

'''
Check that all gates are uniatry
'''


def check_unitary():
    return


'''
Check Toffoli identity
'''


def check_toffoli():
    return
