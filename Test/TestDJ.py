import sys

sys.path.append('..')
import Algorithm

alg = Algorithm.DuetchJosa(2)
uf = [0, 0]
alg.set_input(uf)
alg.construct_circuit()
alg.circuit.Debug = True
alg.compute_result()
