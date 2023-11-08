import sys

import numpy as np
import random

sys.path.append('..')
import Algorithm
from Parameter import maximumqubit
import unittest


def generate_random_balance(qubit_num: int) -> np.ndarray:
    n = (1 << (qubit_num - 1))
    uf = [0] * n
    pos = 0
    for i in range(0, n):
        dice = 0
        while dice < 5:
            dice = random.randint(0, 5)
            if dice > 4:
                uf.insert(pos, 1)
            pos = (pos + 1) % len(uf)
    return uf


class TestDJ(unittest.TestCase):
    def test_constant(self):
        for i in range(2, maximumqubit + 1):
            inputsize = i - 1
            n = (1 << (inputsize))
            dice = random.randint(0, 1)
            uf = [dice] * n
            djalg = Algorithm.DuetchJosa(i)
            djalg.set_input(uf)
            djalg.construct_circuit()
            djalg.compute_result()
            self.assertEqual(djalg.is_balance(), False)

    def test_balance(self):
        for i in range(2, maximumqubit + 1):
            uf = generate_random_balance(i-1)
            djalg = Algorithm.DuetchJosa(i)
            djalg.set_input(uf)
            djalg.construct_circuit()
            djalg.compute_result()
            self.assertEqual(djalg.is_balance(), True)


def main():
    unittest.main()

'''
Example:
    alg = Algorithm.DuetchJosa(4)
    uf = [1, 1, 1, 1, 1, 1, 1, 1]
    alg.set_input(uf)
    alg.construct_circuit()
    alg.compute_result()
    alg.circuit.state.show_state_dirac()
'''


if __name__ == "__main__":
    main()
