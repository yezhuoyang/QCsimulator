import sys

sys.path.append('..')
import Algorithm

alg = Algorithm.BVAlgorithm(9)
a=0b10101001
b=1
alg.set_input([a,b])
alg.construct_circuit()
alg.compute_result()
alg.circuit.state.show_state_dirac()