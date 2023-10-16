
import sys
sys.path.append('..')
import Parameter
import State
import numpy as np


S=State.QuantumState(np.array([1j,0],dtype=Parameter.qtype))
S1=State.QuantumState(np.array([1j,0],dtype=Parameter.qtype))
S2=S.tensor_product(S1)


S2.show_state()


print(abs(S2.state_vector[int("01", 2)])**2)

