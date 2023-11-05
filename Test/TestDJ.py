import sys

sys.path.append('..')
import Algorithm

alg = Algorithm.DuetchJosa(3)
uf = [1,1,1,1]
alg.set_input(uf)
alg.construct_circuit()
alg.circuit.Debug = True
alg.compute_result()
