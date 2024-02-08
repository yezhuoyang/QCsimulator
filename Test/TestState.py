
import sys
sys.path.append('..')
import Parameter
import State
import numpy as np


import State, Parameter
import numpy as np

S=State.QuantumState(np.array([0.7j,0.7],dtype=Parameter.qtype))
S.normalize()
S.show_state_dirac()
