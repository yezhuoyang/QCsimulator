
import sys
sys.path.append('..')
import Parameter
import State
import numpy as np


S=State.QuantumState(np.array([1j,0],dtype=Parameter.qtype))
S.show_state()


