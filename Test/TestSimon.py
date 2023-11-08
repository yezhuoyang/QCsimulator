import sys

sys.path.append('..')
import Algorithm

alg = Algorithm.DuetchJosa(4)
uf = [1,1,1,1,1,1,1,1]
alg.set_input(uf)
alg.construct_circuit()
alg.compute_result()
alg.circuit.state.show_state_dirac()





