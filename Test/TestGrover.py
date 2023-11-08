import sys

import numpy as np
import random

sys.path.append('..')
import Algorithm
from Parameter import maximumqubit
import unittest











if __name__ == "__main__":
    alg = Algorithm.Grover(4)
    b = 1
    alg.set_input([a, b])
    alg.construct_circuit()
    alg.compute_result()
    alg.circuit.state.show_state_dirac()